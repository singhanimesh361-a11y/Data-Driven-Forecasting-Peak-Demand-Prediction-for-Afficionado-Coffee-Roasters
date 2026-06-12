"""Facebook Prophet forecaster for the Afficionado Demand Intelligence Platform.

Wraps ``prophet.Prophet`` with sensible defaults for specialty-coffee retail
demand forecasting.  Key design choices:

* **Weekly seasonality** is always enabled (coffee demand has clear
  day-of-week patterns).
* **Daily (intra-day) seasonality** is enabled only when ``freq='H'``
  (hourly data).
* **Yearly seasonality** is explicitly **disabled** because ADIP typically
  has only ~1 year of data — fitting an annual Fourier series on one cycle
  leads to overfitting.
* ``seasonality_mode='multiplicative'`` captures the fact that seasonal
  swings scale with trend level in revenue data.

Typical usage::

    forecaster = ProphetForecaster(freq='D')
    forecaster.fit(daily_store_df, target_col='total_revenue')
    preds = forecaster.predict_all_stores(horizon=30)
    components = forecaster.get_components(store_id='downtown')
"""

from __future__ import annotations

import io
import logging
import warnings
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base (standalone mirror of BaseForecaster)
# ---------------------------------------------------------------------------
class BaseForecaster(ABC):
    """Minimal abstract base for all ADIP forecasters.

    Full version is in ``src.models.baseline``.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs: Any) -> "BaseForecaster":
        """Train the model."""

    @abstractmethod
    def predict(self, horizon: int, store_id: str, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""


# ---------------------------------------------------------------------------
# MLflow helper stubs
# ---------------------------------------------------------------------------
def _mlflow_log_param(key: str, value: Any) -> None:
    """Log a parameter to MLflow if an active run exists.

    Args:
        key: Parameter name.
        value: Parameter value.
    """
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_param(key, value)
    except ImportError:
        logger.debug("MLflow not installed — skipping log_param(%s)", key)
    except Exception:
        logger.debug("MLflow log_param failed for %s", key, exc_info=True)


def _mlflow_log_metric(key: str, value: float) -> None:
    """Log a metric to MLflow if an active run exists.

    Args:
        key: Metric name.
        value: Numeric metric value.
    """
    try:
        import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_metric(key, value)
    except ImportError:
        logger.debug("MLflow not installed — skipping log_metric(%s)", key)
    except Exception:
        logger.debug("MLflow log_metric failed for %s", key, exc_info=True)


# ---------------------------------------------------------------------------
# Helper: compute MAPE safely
# ---------------------------------------------------------------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zeros in actuals.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (0–100+).  Returns ``float('inf')`` if all
        actuals are zero.
    """
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Prophet Forecaster
# ---------------------------------------------------------------------------
class ProphetForecaster(BaseForecaster):
    """Demand forecaster backed by Facebook Prophet.

    Attributes:
        freq: ``'D'`` for daily or ``'H'`` for hourly granularity.
        models: Mapping from ``store_id`` to a fitted ``Prophet`` object.
        training_data: Mapping from ``store_id`` to the training DataFrame.
        training_mape: Mapping from ``store_id`` to in-sample MAPE.
    """

    def __init__(self, freq: str = "D", n_optuna_trials: int = 0) -> None:
        """Create a new ProphetForecaster.

        Args:
            freq: Temporal frequency — ``'D'`` (daily) or ``'H'`` (hourly).
            n_optuna_trials: Number of optuna trials for hyperparameter tuning.

        Raises:
            ValueError: If *freq* is not ``'D'`` or ``'H'``.
        """
        if freq not in {"D", "H"}:
            raise ValueError(f"freq must be 'D' or 'H', got '{freq}'")

        self.freq: str = freq
        self.n_optuna_trials: int = n_optuna_trials
        self.models: Dict[str, Any] = {}
        self.training_data: Dict[str, pd.DataFrame] = {}
        self.training_mape: Dict[str, float] = {}

    # -- Internal: build a fresh Prophet instance --------------------------
    def _make_prophet(self, **kwargs) -> Any:
        """Construct a Prophet object with ADIP-standard settings.

        Returns:
            An *unfitted* ``Prophet`` instance.
        """
        from prophet import Prophet  # noqa: WPS433 – deferred import

        model = Prophet(
            weekly_seasonality=True,
            daily_seasonality=(self.freq == "H"),
            yearly_seasonality=False,
            uncertainty_samples=1000,
            interval_width=0.80,
            seasonality_mode="multiplicative",
            **kwargs,
        )
        return model

    # -- Fit ---------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = "total_revenue",
        **kwargs: Any,
    ) -> "ProphetForecaster":
        """Fit one Prophet model per store.

        The input DataFrame is expected to have columns ``date``,
        ``store_id``, and *target_col*.  Internally the data is renamed
        to the ``ds`` / ``y`` convention that Prophet requires.

        Prophet's verbose stdout/stderr output is suppressed during fitting.

        Args:
            df: Historical data.
            target_col: Name of the target column.
            **kwargs: Additional keyword arguments (reserved).

        Returns:
            ``self``, for method chaining.

        Raises:
            ValueError: If *target_col* is missing from *df*.
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found. " f"Available: {list(df.columns)}")

        store_ids: List[str] = sorted(df["store_id"].unique().tolist())
        logger.info("Fitting Prophet models for %d store(s): %s", len(store_ids), store_ids)

        _mlflow_log_param("prophet_freq", self.freq)
        _mlflow_log_param("prophet_weekly_seasonality", True)
        _mlflow_log_param("prophet_daily_seasonality", self.freq == "H")
        _mlflow_log_param("prophet_yearly_seasonality", False)
        _mlflow_log_param("prophet_seasonality_mode", "multiplicative")

        for sid in store_ids:
            store_df = df.loc[df["store_id"] == sid].sort_values("date").reset_index(drop=True)
            prophet_df = store_df.rename(columns={"date": "ds", target_col: "y"})[["ds", "y"]].copy()
            prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

            self.training_data[sid] = prophet_df.copy()

            best_params = {}
            if self.n_optuna_trials > 0 and len(prophet_df) > 14:
                import optuna

                optuna.logging.set_verbosity(optuna.logging.ERROR)

                # Time-based split
                val_size = 7
                train_df = prophet_df.iloc[:-val_size]
                val_df = prophet_df.iloc[-val_size:]

                def objective(trial):
                    cps = trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True)
                    sps = trial.suggest_float("seasonality_prior_scale", 0.01, 10.0, log=True)

                    m = self._make_prophet(changepoint_prior_scale=cps, seasonality_prior_scale=sps)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        _devnull = io.StringIO()
                        with redirect_stdout(_devnull), redirect_stderr(_devnull):
                            m.fit(train_df)

                    preds = m.predict(val_df[["ds"]])
                    return _safe_mape(val_df["y"].values, preds["yhat"].values)

                study = optuna.create_study(direction="minimize")
                study.optimize(objective, n_trials=self.n_optuna_trials)
                best_params = study.best_params
                logger.info(f"  [{sid}] Best Prophet params: {best_params}")

            model = self._make_prophet(**best_params)

            logger.info("  [%s] Fitting final Prophet on %d rows …", sid, len(prophet_df))

            # Suppress Prophet's noisy console output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _devnull = io.StringIO()
                with redirect_stdout(_devnull), redirect_stderr(_devnull):
                    model.fit(prophet_df)

            self.models[sid] = model

            # In-sample MAPE
            in_sample = model.predict(prophet_df[["ds"]])
            mape = _safe_mape(prophet_df["y"].values, in_sample["yhat"].values)
            self.training_mape[sid] = mape
            logger.info("  [%s] Training MAPE = %.2f%%", sid, mape)
            _mlflow_log_metric(f"prophet_{sid}_train_mape", mape)

        return self

    # -- Predict -----------------------------------------------------------
    def predict(
        self,
        horizon: int,
        store_id: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate forecasts for a single store.

        Negative predictions are clipped to zero.

        Args:
            horizon: Number of future periods (days or hours) to forecast.
            store_id: Store identifier.

        Returns:
            DataFrame with the ADIP standard forecast schema:
            ``forecast_date``, ``store_id``, ``predicted_value``,
            ``lower_80``, ``upper_80``, ``lower_95``, ``upper_95``,
            ``model_name``.

        Raises:
            KeyError: If *store_id* has not been fitted.
        """
        if store_id not in self.models:
            raise KeyError(f"No fitted model for store_id='{store_id}'. " f"Available: {list(self.models.keys())}")

        model = self.models[store_id]

        future = model.make_future_dataframe(
            periods=horizon,
            freq=self.freq,
            include_history=False,
        )
        forecast_80 = model.predict(future)

        # Prophet's default interval_width is set to 0.80 during init.
        # For 95 % intervals we refit interval width temporarily.
        model_95 = self.models[store_id]
        original_width = model_95.interval_width
        model_95.interval_width = 0.95
        model_95.uncertainty_samples = 1000

        _devnull = io.StringIO()
        with redirect_stdout(_devnull), redirect_stderr(_devnull):
            forecast_95 = model_95.predict(future)
        model_95.interval_width = original_width  # restore

        result = pd.DataFrame(
            {
                "forecast_date": forecast_80["ds"].values,
                "store_id": store_id,
                "predicted_value": np.maximum(forecast_80["yhat"].values, 0.0),
                "lower_80": forecast_80["yhat_lower"].values,
                "upper_80": forecast_80["yhat_upper"].values,
                "lower_95": forecast_95["yhat_lower"].values,
                "upper_95": forecast_95["yhat_upper"].values,
                "model_name": "Prophet",
            }
        )
        return result

    # -- Predict all stores ------------------------------------------------
    def predict_all_stores(self, horizon: int) -> pd.DataFrame:
        """Forecast for every fitted store and concatenate.

        Args:
            horizon: Number of future periods to forecast.

        Returns:
            Concatenated prediction DataFrame across all stores.
        """
        frames: List[pd.DataFrame] = []
        for sid in sorted(self.models.keys()):
            frames.append(self.predict(horizon=horizon, store_id=sid))
        return pd.concat(frames, ignore_index=True)

    # -- Components --------------------------------------------------------
    def get_components(self, store_id: str) -> pd.DataFrame:
        """Extract decomposed components (trend, weekly, daily).

        Args:
            store_id: Store whose model to decompose.

        Returns:
            DataFrame with columns ``ds``, ``trend``, ``weekly``, and
            optionally ``daily`` (only present when ``freq='H'``).

        Raises:
            KeyError: If *store_id* has not been fitted.
        """
        if store_id not in self.models:
            raise KeyError(f"No fitted model for store_id='{store_id}'. " f"Available: {list(self.models.keys())}")

        model = self.models[store_id]
        train_df = self.training_data[store_id]

        _devnull = io.StringIO()
        with redirect_stdout(_devnull), redirect_stderr(_devnull):
            forecast = model.predict(train_df[["ds"]])

        cols = ["ds", "trend", "weekly"]
        if "daily" in forecast.columns:
            cols.append("daily")

        return forecast[cols].copy()

    # -- Repr --------------------------------------------------------------
    def __repr__(self) -> str:
        fitted = list(self.models.keys())
        return f"ProphetForecaster(freq='{self.freq}', stores_fitted={fitted})"


# ---------------------------------------------------------------------------
# Inline unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    def _make_synthetic(
        n_days: int = 120,
        n_stores: int = 2,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic daily revenue data.

        Args:
            n_days: Number of days to simulate.
            n_stores: Number of stores.
            seed: Random seed.

        Returns:
            DataFrame with ``date``, ``store_id``, ``total_revenue``.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_stores):
            dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
            trend = np.linspace(800, 1200, n_days)
            weekly = 150 * np.sin(2 * np.pi * np.arange(n_days) / 7)
            noise = rng.normal(0, 40, n_days)
            revenue = trend + weekly + noise + s * 200
            for d, r in zip(dates, revenue):
                rows.append({"date": d, "store_id": f"store_{s}", "total_revenue": r})
        return pd.DataFrame(rows)

    def test_fit_daily() -> None:
        """Prophet should fit daily data without error."""
        df = _make_synthetic(n_days=120, n_stores=1)
        fc = ProphetForecaster(freq="D")
        fc.fit(df)
        assert "store_0" in fc.models
        assert fc.training_mape["store_0"] < 100
        print("PASS: test_fit_daily")

    def test_non_negative() -> None:
        """Point predictions must be ≥ 0."""
        df = _make_synthetic(n_days=120, n_stores=1)
        fc = ProphetForecaster(freq="D")
        fc.fit(df)
        preds = fc.predict(horizon=14, store_id="store_0")
        assert (preds["predicted_value"] >= 0).all()
        print("PASS: test_non_negative")

    def test_schema() -> None:
        """Output DataFrame must follow the ADIP prediction schema."""
        df = _make_synthetic(n_days=120, n_stores=1)
        fc = ProphetForecaster(freq="D")
        fc.fit(df)
        preds = fc.predict(horizon=7, store_id="store_0")
        expected_cols = {
            "forecast_date",
            "store_id",
            "predicted_value",
            "lower_80",
            "upper_80",
            "lower_95",
            "upper_95",
            "model_name",
        }
        assert set(preds.columns) == expected_cols
        assert len(preds) == 7
        print("PASS: test_schema")

    def test_components() -> None:
        """get_components should return trend and weekly columns."""
        df = _make_synthetic(n_days=120, n_stores=1)
        fc = ProphetForecaster(freq="D")
        fc.fit(df)
        comps = fc.get_components(store_id="store_0")
        assert "trend" in comps.columns
        assert "weekly" in comps.columns
        print("PASS: test_components")

    test_fit_daily()
    test_non_negative()
    test_schema()
    test_components()
    print("\nAll Prophet inline tests passed ✓")
