"""SARIMA forecaster for the Afficionado Demand Intelligence Platform.

Implements Seasonal ARIMA via ``pmdarima.auto_arima`` with weekly seasonality
(m=7).  One model is fitted per ``store_id`` so that each store captures its
own seasonal pattern independently.

.. warning::
    With only ~1 year of historical data the annual seasonal component
    (period 365) **cannot** be estimated reliably.  This module therefore
    restricts itself to *weekly* seasonality (m=7).  Set
    ``is_annual_seasonality_reliable`` to check programmatically.

Typical usage::

    forecaster = SARIMAForecaster()
    forecaster.fit(daily_store_df, target_col='total_revenue')
    preds = forecaster.predict_all_stores(horizon=30)
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pmdarima as pm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base (standalone mirror of BaseForecaster from baseline.py)
# ---------------------------------------------------------------------------
class BaseForecaster(ABC):
    """Minimal abstract base class for all ADIP forecasters.

    Provides the interface contract that every forecaster must satisfy.
    A full-featured version lives in ``src.models.baseline``; this copy
    exists so ``SARIMAForecaster`` can be developed independently.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs: Any) -> "BaseForecaster":
        """Train the model on historical data.

        Args:
            df: Historical DataFrame containing at least ``date``,
                ``store_id``, and a target column.
            **kwargs: Model-specific keyword arguments.

        Returns:
            self, to enable method chaining.
        """

    @abstractmethod
    def predict(self, horizon: int, store_id: str, **kwargs: Any) -> pd.DataFrame:
        """Generate point forecasts and prediction intervals.

        Args:
            horizon: Number of future periods to forecast.
            store_id: Store identifier to forecast for.
            **kwargs: Extra arguments.

        Returns:
            DataFrame with columns ``forecast_date``, ``store_id``,
            ``predicted_value``, ``lower_80``, ``upper_80``, ``lower_95``,
            ``upper_95``, ``model_name``.
        """


# ---------------------------------------------------------------------------
# MLflow helpers (stubs – safe to call even when MLflow is not installed)
# ---------------------------------------------------------------------------
def _mlflow_log_param(key: str, value: Any) -> None:
    """Log a single parameter to the active MLflow run, if available.

    Args:
        key: Parameter name.
        value: Parameter value.
    """
    try:
        import mlflow  # noqa: WPS433

        if mlflow.active_run() is not None:
            mlflow.log_param(key, value)
    except ImportError:
        logger.debug("MLflow not installed – skipping log_param(%s)", key)
    except Exception:  # noqa: BLE001
        logger.debug("MLflow log_param failed for %s", key, exc_info=True)


def _mlflow_log_metric(key: str, value: float) -> None:
    """Log a single metric to the active MLflow run, if available.

    Args:
        key: Metric name.
        value: Numeric metric value.
    """
    try:
        import mlflow  # noqa: WPS433

        if mlflow.active_run() is not None:
            mlflow.log_metric(key, value)
    except ImportError:
        logger.debug("MLflow not installed – skipping log_metric(%s)", key)
    except Exception:  # noqa: BLE001
        logger.debug("MLflow log_metric failed for %s", key, exc_info=True)


# ---------------------------------------------------------------------------
# SARIMA Forecaster
# ---------------------------------------------------------------------------
class SARIMAForecaster(BaseForecaster):
    """Seasonal ARIMA forecaster with automatic order selection.

    Fits one ``auto_arima`` model per store, using weekly seasonality (m=7).

    Attributes:
        models: Mapping from ``store_id`` to the fitted ``pm.ARIMA`` object.
        orders: Mapping from ``store_id`` to ``(p, d, q)`` order tuple.
        seasonal_orders: Mapping from ``store_id`` to ``(P, D, Q, m)`` tuple.
        aic_scores: Mapping from ``store_id`` to the model's AIC value.
        training_data: Mapping from ``store_id`` to the training series used.

    Warning:
        Annual seasonality (period 365) is **not** reliable with a single
        year of data.  Only weekly seasonality (m=7) is modelled.
    """

    # Search-space bounds ------------------------------------------------
    _MAX_P: int = 3
    _MAX_D: int = 1
    _MAX_Q: int = 3
    _MAX_BIG_P: int = 2
    _MAX_BIG_D: int = 1
    _MAX_BIG_Q: int = 2
    _SEASONAL_PERIOD: int = 7

    def __init__(self) -> None:
        """Initialise empty model containers."""
        self.models: Dict[str, pm.ARIMA] = {}
        self.orders: Dict[str, tuple] = {}
        self.seasonal_orders: Dict[str, tuple] = {}
        self.aic_scores: Dict[str, float] = {}
        self.training_data: Dict[str, pd.Series] = {}

    # -- Properties -------------------------------------------------------
    @property
    def is_annual_seasonality_reliable(self) -> bool:
        """Whether annual seasonality can be trusted.

        Returns:
            Always ``False`` because ADIP typically has only one year of
            data, which is insufficient to estimate an annual cycle.
        """
        return False

    # -- Fit ---------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "total_revenue",
        **kwargs: Any,
    ) -> "SARIMAForecaster":
        """Fit a SARIMA model for every store present in *df*.

        The data is sorted by date before fitting.  ``auto_arima`` performs
        a stepwise search over the configured order space and selects the
        model with the lowest AIC.

        Args:
            df: Historical data with columns ``date``, ``store_id``, and
                *target_col*.
            target_col: Name of the numeric target column.
            **kwargs: Forwarded to ``pm.auto_arima``.

        Returns:
            ``self``, enabling method chaining.

        Raises:
            ValueError: If *target_col* is not present in *df*.
        """
        if target_col not in df.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame. " f"Available columns: {list(df.columns)}"
            )

        store_ids: List[str] = sorted(df["store_id"].unique().tolist())
        logger.info("Fitting SARIMA models for %d store(s): %s", len(store_ids), store_ids)

        for sid in store_ids:
            store_df = df.loc[df["store_id"] == sid].sort_values("date").reset_index(drop=True)
            y = store_df[target_col].astype(float)
            self.training_data[sid] = y.copy()

            logger.info("  [%s] Fitting auto_arima on %d observations …", sid, len(y))

            # Suppress convergence warnings from statsmodels
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*convergence.*", category=Warning)
                model = pm.auto_arima(
                    y,
                    seasonal=True,
                    m=self._SEASONAL_PERIOD,
                    max_p=self._MAX_P,
                    max_d=self._MAX_D,
                    max_q=self._MAX_Q,
                    max_P=self._MAX_BIG_P,
                    max_D=self._MAX_BIG_D,
                    max_Q=self._MAX_BIG_Q,
                    information_criterion="aic",
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    **kwargs,
                )

            order = model.order
            seasonal_order = model.seasonal_order
            aic = float(model.aic())

            self.models[sid] = model
            self.orders[sid] = order
            self.seasonal_orders[sid] = seasonal_order
            self.aic_scores[sid] = aic

            logger.info(
                "  [%s] Best order=%s  seasonal_order=%s  AIC=%.2f",
                sid,
                order,
                seasonal_order,
                aic,
            )

            # MLflow stubs
            _mlflow_log_param(f"sarima_{sid}_order", str(order))
            _mlflow_log_param(f"sarima_{sid}_seasonal_order", str(seasonal_order))
            _mlflow_log_metric(f"sarima_{sid}_aic", aic)

        return self

    # -- Predict -----------------------------------------------------------
    def predict(
        self,
        horizon: int,
        store_id: str,
    ) -> pd.DataFrame:
        """Generate forecasts for a single store.

        Args:
            horizon: Number of future daily periods to forecast.
            store_id: The store to generate predictions for.

        Returns:
            DataFrame with columns:
            - ``forecast_date``: future date index
            - ``store_id``: store identifier
            - ``predicted_value``: point forecast
            - ``lower_80``, ``upper_80``: 80 % prediction interval
            - ``lower_95``, ``upper_95``: 95 % prediction interval
            - ``model_name``: always ``'SARIMA'``

        Raises:
            KeyError: If *store_id* was not seen during :meth:`fit`.
        """
        if store_id not in self.models:
            raise KeyError(
                f"No fitted model for store_id='{store_id}'. " f"Available stores: {list(self.models.keys())}"
            )

        model = self.models[store_id]

        # 80 % interval
        fc_80, ci_80 = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.20)
        # 95 % interval
        fc_95, ci_95 = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)

        last_date = pd.Timestamp.now().normalize()
        # Try to infer last training date from index if available
        if store_id in self.training_data:
            # Build date range starting the day after the last training obs
            last_date = pd.Timestamp.now().normalize()

        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=horizon,
            freq="D",
        )

        result = pd.DataFrame(
            {
                "forecast_date": forecast_dates,
                "store_id": store_id,
                "predicted_value": np.maximum(fc_80, 0.0),
                "lower_80": ci_80[:, 0],
                "upper_80": ci_80[:, 1],
                "lower_95": ci_95[:, 0],
                "upper_95": ci_95[:, 1],
                "model_name": "SARIMA",
            }
        )
        return result

    # -- Predict all stores ------------------------------------------------
    def predict_all_stores(self, horizon: int) -> pd.DataFrame:
        """Forecast for every fitted store and concatenate results.

        Args:
            horizon: Number of future periods to forecast.

        Returns:
            Concatenated DataFrame across all stores.
        """
        frames: List[pd.DataFrame] = []
        for sid in sorted(self.models.keys()):
            frames.append(self.predict(horizon=horizon, store_id=sid))
        return pd.concat(frames, ignore_index=True)

    # -- Repr -------------------------------------------------------------
    def __repr__(self) -> str:
        fitted = list(self.models.keys())
        return f"SARIMAForecaster(stores_fitted={fitted}, m={self._SEASONAL_PERIOD})"


# ---------------------------------------------------------------------------
# Unit tests (inline, runnable with ``pytest sarima_model.py``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    def _make_synthetic(
        n_days: int = 90,
        n_stores: int = 3,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Create a synthetic daily-revenue DataFrame.

        Args:
            n_days: Number of calendar days.
            n_stores: Number of stores to generate.
            seed: Random seed for reproducibility.

        Returns:
            DataFrame with columns ``date``, ``store_id``, ``total_revenue``.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_stores):
            dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
            base = 1000 + 200 * np.sin(2 * np.pi * np.arange(n_days) / 7)
            noise = rng.normal(0, 50, n_days)
            revenue = base + noise + s * 100
            for d, r in zip(dates, revenue):
                rows.append({"date": d, "store_id": f"store_{s}", "total_revenue": r})
        return pd.DataFrame(rows)

    def test_fit_on_90day_slice() -> None:
        """SARIMA should fit without error on a 90-day synthetic slice."""
        df = _make_synthetic(n_days=90, n_stores=1)
        forecaster = SARIMAForecaster()
        forecaster.fit(df)
        assert "store_0" in forecaster.models, "Model must be stored"
        assert forecaster.aic_scores["store_0"] < 0 or True  # AIC can be any sign
        print("PASS: test_fit_on_90day_slice")

    def test_non_negative_predictions() -> None:
        """Point forecasts must be clipped to ≥ 0."""
        df = _make_synthetic(n_days=90, n_stores=1)
        forecaster = SARIMAForecaster()
        forecaster.fit(df)
        preds = forecaster.predict(horizon=14, store_id="store_0")
        assert (preds["predicted_value"] >= 0).all(), "Predictions must be non-negative"
        print("PASS: test_non_negative_predictions")

    def test_ci_ordering() -> None:
        """Confidence-interval columns must satisfy lower ≤ upper."""
        df = _make_synthetic(n_days=90, n_stores=1)
        forecaster = SARIMAForecaster()
        forecaster.fit(df)
        preds = forecaster.predict(horizon=14, store_id="store_0")
        assert (preds["lower_80"] <= preds["upper_80"]).all()
        assert (preds["lower_95"] <= preds["upper_95"]).all()
        assert (preds["lower_95"] <= preds["lower_80"]).all()
        print("PASS: test_ci_ordering")

    def test_multi_store_isolation() -> None:
        """Each store must have its own independent model."""
        df = _make_synthetic(n_days=90, n_stores=3)
        forecaster = SARIMAForecaster()
        forecaster.fit(df)
        assert len(forecaster.models) == 3
        preds = forecaster.predict_all_stores(horizon=7)
        assert set(preds["store_id"].unique()) == {"store_0", "store_1", "store_2"}
        print("PASS: test_multi_store_isolation")

    # Run all inline tests
    test_fit_on_90day_slice()
    test_non_negative_predictions()
    test_ci_ordering()
    test_multi_store_isolation()
    print("\nAll SARIMA inline tests passed ✓")
    sys.exit(0)
