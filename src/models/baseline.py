"""Baseline forecasting models for ADIP demand intelligence.

Provides simple, interpretable forecasters (Naive, Seasonal Naive,
Moving Average) that serve as benchmarks for more complex models.
All forecasters follow the BaseForecaster ABC contract.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base class for all ADIP forecasters.

    Defines the fit / predict / predict_with_ci contract that every
    forecaster must implement.  Subclasses are expected to be
    pickle-serializable so they can be persisted to the model registry.

    Attributes:
        model_name: Human-readable model identifier.
        _is_fitted: Whether fit() has been called successfully.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._is_fitted: bool = False

    @abstractmethod
    def fit(self, df: pd.DataFrame, target_col: str = "demand") -> "BaseForecaster":
        """Fit the forecaster on historical data.

        Args:
            df: Historical DataFrame with at least ``store_id``, ``date``,
                and ``target_col`` columns.
            target_col: Name of the column to forecast.

        Returns:
            self, for method chaining.
        """
        ...

    @abstractmethod
    def predict(self, horizon: int, store_ids: Optional[list[str]] = None) -> pd.DataFrame:
        """Generate point forecasts for *horizon* future steps.

        Args:
            horizon: Number of future periods to forecast.
            store_ids: Optional subset of store IDs.  If ``None`` all
                fitted stores are used.

        Returns:
            DataFrame with columns: ``forecast_date``, ``store_id``,
            ``predicted_value``, ``lower_80``, ``upper_80``, ``lower_95``,
            ``upper_95``, ``model_name``.
        """
        ...

    @abstractmethod
    def predict_with_ci(
        self,
        horizon: int,
        store_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Generate forecasts with confidence intervals.

        Uses historical residual standard deviation for interval width:
        ±1.28σ for 80 % CI, ±1.96σ for 95 % CI.

        Args:
            horizon: Number of future periods to forecast.
            store_ids: Optional subset of store IDs.

        Returns:
            DataFrame identical in schema to ``predict()``.
        """
        ...

    def _validate_fitted(self) -> None:
        """Raise if the model has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(f"{self.model_name} must be fit before predict.")

    def _build_forecast_df(
        self,
        store_id: str,
        values: np.ndarray,
        start_date: pd.Timestamp,
        residual_std: float,
    ) -> pd.DataFrame:
        """Helper to build a standardised forecast DataFrame for one store.

        Args:
            store_id: Store identifier.
            values: Array of predicted values with length == horizon.
            start_date: First forecast date (day after last training date).
            residual_std: Standard deviation of in-sample residuals.

        Returns:
            DataFrame with the canonical forecast columns.
        """
        horizon = len(values)
        dates = pd.date_range(start=start_date, periods=horizon, freq="D")
        return pd.DataFrame(
            {
                "forecast_date": dates,
                "store_id": store_id,
                "predicted_value": values,
                "lower_80": values - 1.28 * residual_std,
                "upper_80": values + 1.28 * residual_std,
                "lower_95": values - 1.96 * residual_std,
                "upper_95": values + 1.96 * residual_std,
                "model_name": self.model_name,
            }
        )

    # ---- Pickle serialization support ----
    def __getstate__(self) -> dict:
        return self.__dict__.copy()

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# Naive Forecaster
# ---------------------------------------------------------------------------


class NaiveForecaster(BaseForecaster):
    """Repeats the last observed value for every horizon step.

    The simplest possible baseline: for each store the forecast equals the
    final training observation repeated *horizon* times.

    Attributes:
        _last_values: Mapping ``store_id -> last observed target value``.
        _residual_stds: Mapping ``store_id -> residual standard deviation``.
        _last_dates: Mapping ``store_id -> last training date``.
    """

    def __init__(self) -> None:
        super().__init__(model_name="NaiveForecaster")
        self._last_values: dict[str, float] = {}
        self._residual_stds: dict[str, float] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    def fit(self, df: pd.DataFrame, target_col: str = "demand") -> "NaiveForecaster":
        """Fit by storing the last value and residual std per store.

        Args:
            df: Historical DataFrame.
            target_col: Target column name.

        Returns:
            self
        """
        logger.debug("NaiveForecaster.fit – %d rows", len(df))
        for store_id, grp in df.groupby("store_id"):
            grp = grp.sort_values("date")
            values = grp[target_col].values.astype(float)
            self._last_values[store_id] = values[-1]
            self._last_dates[store_id] = pd.to_datetime(grp["date"].iloc[-1])
            # Residual = difference between consecutive values (naive residuals)
            if len(values) > 1:
                residuals = np.diff(values)
                self._residual_stds[store_id] = float(np.std(residuals, ddof=1))
            else:
                self._residual_stds[store_id] = 0.0
        self._is_fitted = True
        logger.debug("NaiveForecaster fitted for %d stores", len(self._last_values))
        return self

    def predict(self, horizon: int, store_ids: Optional[list[str]] = None) -> pd.DataFrame:
        """Repeat the last observed value for *horizon* days.

        Args:
            horizon: Forecast horizon in days.
            store_ids: Optional store subset.

        Returns:
            Forecast DataFrame.
        """
        return self.predict_with_ci(horizon, store_ids)

    def predict_with_ci(
        self,
        horizon: int,
        store_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Naive forecast with confidence intervals.

        Args:
            horizon: Forecast horizon in days.
            store_ids: Optional store subset.

        Returns:
            Forecast DataFrame with CI columns.
        """
        self._validate_fitted()
        store_ids = store_ids or list(self._last_values.keys())
        logger.debug("NaiveForecaster.predict – horizon=%d, stores=%d", horizon, len(store_ids))

        frames: list[pd.DataFrame] = []
        for sid in store_ids:
            last_val = self._last_values[sid]
            pred = np.full(horizon, last_val)
            start = self._last_dates[sid] + pd.Timedelta(days=1)
            frames.append(self._build_forecast_df(sid, pred, start, self._residual_stds[sid]))
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Seasonal Naive Forecaster
# ---------------------------------------------------------------------------


class SeasonalNaiveForecaster(BaseForecaster):
    """Repeats the last same-weekday value in a weekly cycle.

    For each store, stores the last 7 same-weekday observations and cycles
    through them for the forecast horizon.

    Attributes:
        _weekday_values: ``store_id -> {weekday_int: last_value}``.
        _residual_stds: ``store_id -> residual std``.
        _last_dates: ``store_id -> last training date``.
    """

    def __init__(self) -> None:
        super().__init__(model_name="SeasonalNaiveForecaster")
        self._weekday_values: dict[str, dict[int, float]] = {}
        self._residual_stds: dict[str, float] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    def fit(self, df: pd.DataFrame, target_col: str = "demand") -> "SeasonalNaiveForecaster":
        """Fit by recording the last value for each weekday per store.

        Args:
            df: Historical DataFrame.
            target_col: Target column name.

        Returns:
            self
        """
        logger.debug("SeasonalNaiveForecaster.fit – %d rows", len(df))
        for store_id, grp in df.groupby("store_id"):
            grp = grp.sort_values("date").copy()
            grp["_dt"] = pd.to_datetime(grp["date"])
            grp["_dow"] = grp["_dt"].dt.dayofweek

            weekday_vals: dict[int, float] = {}
            for dow in range(7):
                subset = grp.loc[grp["_dow"] == dow, target_col]
                if len(subset) > 0:
                    weekday_vals[dow] = float(subset.iloc[-1])
            self._weekday_values[store_id] = weekday_vals
            self._last_dates[store_id] = grp["_dt"].max()

            # Seasonal residual: value[t] - value[t-7]
            values = grp[target_col].values.astype(float)
            if len(values) > 7:
                residuals = values[7:] - values[:-7]
                self._residual_stds[store_id] = float(np.std(residuals, ddof=1))
            else:
                self._residual_stds[store_id] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        self._is_fitted = True
        logger.debug("SeasonalNaiveForecaster fitted for %d stores", len(self._weekday_values))
        return self

    def predict(self, horizon: int, store_ids: Optional[list[str]] = None) -> pd.DataFrame:
        """Seasonal-naive point forecast.

        Args:
            horizon: Forecast horizon.
            store_ids: Optional store subset.

        Returns:
            Forecast DataFrame.
        """
        return self.predict_with_ci(horizon, store_ids)

    def predict_with_ci(
        self,
        horizon: int,
        store_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Seasonal-naive forecast with confidence intervals.

        Args:
            horizon: Forecast horizon.
            store_ids: Optional store subset.

        Returns:
            Forecast DataFrame with CI columns.
        """
        self._validate_fitted()
        store_ids = store_ids or list(self._weekday_values.keys())
        logger.debug(
            "SeasonalNaiveForecaster.predict – horizon=%d, stores=%d",
            horizon,
            len(store_ids),
        )

        frames: list[pd.DataFrame] = []
        for sid in store_ids:
            start = self._last_dates[sid] + pd.Timedelta(days=1)
            dates = pd.date_range(start=start, periods=horizon, freq="D")
            preds = np.array([self._weekday_values[sid].get(d.dayofweek, 0.0) for d in dates])
            frames.append(self._build_forecast_df(sid, preds, start, self._residual_stds[sid]))
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Moving Average Forecaster
# ---------------------------------------------------------------------------


class MovingAverageForecaster(BaseForecaster):
    """Rolling-mean forecast that iteratively updates predictions.

    For each horizon step the forecaster computes the mean of the last
    ``window`` observations (including previously predicted values) and
    uses that as the next prediction.

    Args:
        window: Number of trailing observations to average.

    Attributes:
        window: Rolling-window size.
        _histories: ``store_id -> np.ndarray`` of trailing values.
        _residual_stds: ``store_id -> residual std``.
        _last_dates: ``store_id -> last training date``.
    """

    def __init__(self, window: int = 7) -> None:
        super().__init__(model_name="MovingAverageForecaster")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._histories: dict[str, np.ndarray] = {}
        self._residual_stds: dict[str, float] = {}
        self._last_dates: dict[str, pd.Timestamp] = {}

    def fit(self, df: pd.DataFrame, target_col: str = "demand") -> "MovingAverageForecaster":
        """Fit by storing the last ``window`` values and residual std.

        Args:
            df: Historical DataFrame.
            target_col: Target column name.

        Returns:
            self
        """
        logger.debug("MovingAverageForecaster.fit – %d rows, window=%d", len(df), self.window)
        for store_id, grp in df.groupby("store_id"):
            grp = grp.sort_values("date")
            values = grp[target_col].values.astype(float)
            self._histories[store_id] = values[-self.window :]
            self._last_dates[store_id] = pd.to_datetime(grp["date"].iloc[-1])

            # In-sample residuals: actual - rolling mean prediction
            if len(values) > self.window:
                rolling_preds = pd.Series(values).rolling(window=self.window, min_periods=1).mean().values
                residuals = values[self.window :] - rolling_preds[self.window - 1 : -1]
                self._residual_stds[store_id] = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
            else:
                self._residual_stds[store_id] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        self._is_fitted = True
        logger.debug("MovingAverageForecaster fitted for %d stores", len(self._histories))
        return self

    def predict(self, horizon: int, store_ids: Optional[list[str]] = None) -> pd.DataFrame:
        """Moving-average point forecast.

        Args:
            horizon: Forecast horizon.
            store_ids: Optional store subset.

        Returns:
            Forecast DataFrame.
        """
        return self.predict_with_ci(horizon, store_ids)

    def predict_with_ci(
        self,
        horizon: int,
        store_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Moving-average forecast with confidence intervals.

        Each step's prediction is the mean of the last ``window``
        values (including prior predictions within the horizon).

        Args:
            horizon: Forecast horizon.
            store_ids: Optional store subset.

        Returns:
            Forecast DataFrame with CI columns.
        """
        self._validate_fitted()
        store_ids = store_ids or list(self._histories.keys())
        logger.debug(
            "MovingAverageForecaster.predict – horizon=%d, stores=%d",
            horizon,
            len(store_ids),
        )

        frames: list[pd.DataFrame] = []
        for sid in store_ids:
            history = list(self._histories[sid])
            preds: list[float] = []
            for _ in range(horizon):
                window_slice = history[-self.window :]
                pred = float(np.mean(window_slice))
                preds.append(pred)
                history.append(pred)

            start = self._last_dates[sid] + pd.Timedelta(days=1)
            frames.append(self._build_forecast_df(sid, np.array(preds), start, self._residual_stds[sid]))
        return pd.concat(frames, ignore_index=True)
