"""Ensemble forecaster for the Afficionado Demand Intelligence Platform.

Combines Prophet and XGBoost predictions through a Ridge-regression
meta-learner.  A **quality gate** ensures that only models meeting a
minimum accuracy threshold (MAPE ≤ 25 %) contribute to the ensemble.

Architecture::

    Prophet  ──┐
               ├─► Ridge meta-learner ──► weighted forecast
    XGBoost ──┘

Typical usage::

    ensemble = EnsembleForecaster(prophet_fc, xgb_fc, quality_threshold_mape=25.0)
    ensemble.fit(val_df, feature_cols=['lag_1', 'dow', 'rolling_7'])
    preds = ensemble.predict_all_stores(horizon=30, X_future=X_future)
    weights = ensemble.get_weights()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MLflow stubs
# ---------------------------------------------------------------------------
def _mlflow_log_param(key: str, value: Any) -> None:
    """Log a parameter to the active MLflow run, if available.

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
    """Log a metric to the active MLflow run, if available.

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
# MAPE helper
# ---------------------------------------------------------------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (0–100+), ignoring zero actuals.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        MAPE percentage. ``inf`` if all actuals are zero.
    """
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------
class EnsembleQualityError(Exception):
    """Raised when all component models fail the quality gate.

    This prevents the ensemble from producing forecasts that would be
    based entirely on unreliable sub-models.
    """


# ---------------------------------------------------------------------------
# Ensemble Forecaster
# ---------------------------------------------------------------------------
class EnsembleForecaster:
    """Meta-learner ensemble of Prophet and XGBoost forecasters.

    A Ridge regression is fitted on validation-set predictions from both
    sub-models to learn optimal combination weights.  A quality gate
    excludes any sub-model whose validation MAPE exceeds
    ``quality_threshold_mape``.

    Attributes:
        prophet: The Prophet forecaster instance.
        xgb: The XGBoost forecaster instance.
        quality_threshold_mape: Maximum acceptable MAPE (%) for a sub-model
            to participate in the ensemble.
        meta_learners: Per-store Ridge regression meta-learners.
        model_mapes: Per-store MAPE for each sub-model on validation data.
        active_models: Per-store list of sub-model names that passed the
            quality gate.
    """

    def __init__(
        self,
        prophet: Any,
        xgb: Any,
        lstm: Optional[Any] = None,
        quality_threshold_mape: float = 25.0,
    ) -> None:
        """Initialise the ensemble.

        Args:
            prophet: A fitted ``ProphetForecaster`` instance.
            xgb: A fitted ``XGBoostForecaster`` instance.
            lstm: An optional fitted ``LSTMForecaster`` instance.
            quality_threshold_mape: Models with validation MAPE above this
                value are excluded from the ensemble.
        """
        self.prophet = prophet
        self.xgb = xgb
        self.lstm = lstm
        self.quality_threshold_mape: float = quality_threshold_mape

        self.meta_learners: Dict[str, Ridge] = {}
        self.model_mapes: Dict[str, Dict[str, float]] = {}
        self.active_models: Dict[str, List[str]] = {}
        self._feature_cols: List[str] = []

    # -- Fit ---------------------------------------------------------------
    def fit(
        self,
        val_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "total_revenue",
    ) -> "EnsembleForecaster":
        """Train the Ridge meta-learner on validation-set predictions.

        For each store:

        1. Generate Prophet predictions on the validation dates.
        2. Generate XGBoost predictions on the validation features.
        3. Compute MAPE for each; exclude models exceeding the threshold.
        4. If both fail, raise :class:`EnsembleQualityError`.
        5. Fit a ``Ridge(alpha=1.0)`` on the qualifying predictions.

        Args:
            val_df: Validation DataFrame with ``date``, ``store_id``,
                *feature_cols*, and *target_col*.
            feature_cols: Feature columns for XGBoost.
            target_col: Target column name.

        Returns:
            ``self``, for method chaining.

        Raises:
            EnsembleQualityError: If **both** sub-models exceed the quality
                threshold for any store.
        """
        self._feature_cols = list(feature_cols)
        store_ids: List[str] = sorted(val_df["store_id"].unique().tolist())

        _mlflow_log_param("ensemble_quality_threshold", self.quality_threshold_mape)

        for sid in store_ids:
            store_val = val_df.loc[val_df["store_id"] == sid].sort_values("date").reset_index(drop=True)
            y_true = store_val[target_col].values.astype(float)
            n_val = len(store_val)

            # -- Prophet predictions on validation dates --------------------
            prophet_preds = self._get_prophet_val_preds(store_val, sid, n_val)
            prophet_mape = _safe_mape(y_true, prophet_preds)

            # -- XGBoost predictions on validation features -----------------
            xgb_preds = self._get_xgb_val_preds(store_val, sid, feature_cols)
            xgb_mape = _safe_mape(y_true, xgb_preds)

            # -- LSTM predictions on validation features -----------------
            lstm_preds = self._get_lstm_val_preds(store_val, sid, feature_cols) if self.lstm else None
            lstm_mape = _safe_mape(y_true, lstm_preds) if lstm_preds is not None else None

            self.model_mapes[sid] = {
                "prophet": prophet_mape,
                "xgb": xgb_mape,
            }
            if lstm_mape is not None:
                self.model_mapes[sid]["lstm"] = lstm_mape

            logger.info(
                "  [%s] Prophet MAPE=%.2f%%  XGBoost MAPE=%.2f%%  LSTM MAPE=%s",
                sid,
                prophet_mape,
                xgb_mape,
                f"{lstm_mape:.2f}%" if lstm_mape is not None else "N/A",
            )
            _mlflow_log_metric(f"ensemble_{sid}_prophet_mape", prophet_mape)
            _mlflow_log_metric(f"ensemble_{sid}_xgb_mape", xgb_mape)
            if lstm_mape is not None:
                _mlflow_log_metric(f"ensemble_{sid}_lstm_mape", lstm_mape)

            # -- Quality gate -----------------------------------------------
            active: List[str] = []
            columns: List[np.ndarray] = []

            if prophet_mape <= self.quality_threshold_mape:
                active.append("prophet")
                columns.append(prophet_preds)
            else:
                logger.warning(
                    "  [%s] Prophet excluded (MAPE=%.2f%% > %.2f%%)",
                    sid,
                    prophet_mape,
                    self.quality_threshold_mape,
                )

            if xgb_mape <= self.quality_threshold_mape:
                active.append("xgb")
                columns.append(xgb_preds)
            else:
                logger.warning(
                    "  [%s] XGBoost excluded (MAPE=%.2f%% > %.2f%%)",
                    sid,
                    xgb_mape,
                    self.quality_threshold_mape,
                )

            if lstm_mape is not None:
                if lstm_mape <= self.quality_threshold_mape:
                    active.append("lstm")
                    columns.append(lstm_preds)
                else:
                    logger.warning(
                        "  [%s] LSTM excluded (MAPE=%.2f%% > %.2f%%)",
                        sid,
                        lstm_mape,
                        self.quality_threshold_mape,
                    )

            if not active:
                raise EnsembleQualityError(
                    f"Both sub-models failed the quality gate for store "
                    f"'{sid}'. Prophet MAPE={prophet_mape:.2f}%, "
                    f"XGBoost MAPE={xgb_mape:.2f}%, "
                    f"threshold={self.quality_threshold_mape:.2f}%."
                )

            self.active_models[sid] = active

            # -- Ridge meta-learner -----------------------------------------
            X_meta = np.column_stack(columns)
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_meta, y_true)
            self.meta_learners[sid] = ridge

            logger.info(
                "  [%s] Meta-learner fitted on %s  (weights=%s)",
                sid,
                active,
                ridge.coef_.tolist(),
            )
            _mlflow_log_param(f"ensemble_{sid}_active_models", str(active))

        return self

    # -- Internal helpers ---------------------------------------------------
    def _get_prophet_val_preds(
        self,
        store_val: pd.DataFrame,
        store_id: str,
        n_val: int,
    ) -> np.ndarray:
        """Generate Prophet predictions aligned to the validation set.

        Args:
            store_val: Store-specific validation DataFrame.
            store_id: Store identifier.
            n_val: Number of validation observations.

        Returns:
            1-D numpy array of point predictions.
        """
        try:
            preds_df = self.prophet.predict(horizon=n_val, store_id=store_id)
            return np.maximum(preds_df["predicted_value"].values[:n_val], 0.0)
        except Exception:
            logger.warning(
                "  [%s] Prophet prediction failed — using zeros",
                store_id,
                exc_info=True,
            )
            return np.zeros(n_val)

    def _get_xgb_val_preds(
        self,
        store_val: pd.DataFrame,
        store_id: str,
        feature_cols: List[str],
    ) -> np.ndarray:
        """Generate XGBoost predictions on the validation features.

        Args:
            store_val: Store-specific validation DataFrame.
            store_id: Store identifier.
            feature_cols: Column names to use as features.

        Returns:
            1-D numpy array of point predictions.
        """
        try:
            preds_df = self.xgb.predict(X=store_val, store_id=store_id)
            return np.maximum(preds_df["predicted_value"].values, 0.0)
        except Exception:
            logger.warning(
                "  [%s] XGBoost prediction failed — using zeros",
                store_id,
                exc_info=True,
            )
            return np.zeros(len(store_val))

    def _get_lstm_val_preds(
        self,
        store_val: pd.DataFrame,
        store_id: str,
        feature_cols: List[str],
    ) -> np.ndarray:
        try:
            preds_df = self.lstm.predict(X=store_val, store_id=store_id)
            return np.maximum(preds_df["predicted_value"].values, 0.0)
        except Exception:
            logger.warning("  [%s] LSTM prediction failed — using zeros", store_id, exc_info=True)
            return np.zeros(len(store_val))

    # -- Predict -----------------------------------------------------------
    def predict(
        self,
        horizon: int,
        X_future: pd.DataFrame,
        store_id: str,
    ) -> pd.DataFrame:
        """Generate ensemble predictions for a single store.

        Confidence intervals are computed as a weighted average of the
        sub-model intervals, using the Ridge meta-learner weights.

        Args:
            horizon: Number of future periods to forecast.
            X_future: Future feature DataFrame for XGBoost.
            store_id: Store identifier.

        Returns:
            DataFrame with the ADIP standard prediction schema.

        Raises:
            KeyError: If *store_id* was not fitted.
        """
        if store_id not in self.meta_learners:
            raise KeyError(
                f"No meta-learner for store_id='{store_id}'. " f"Available: {list(self.meta_learners.keys())}"
            )

        active = self.active_models[store_id]
        ridge = self.meta_learners[store_id]

        # Collect sub-model predictions
        sub_preds: Dict[str, pd.DataFrame] = {}
        columns: List[np.ndarray] = []

        if "prophet" in active:
            prophet_df = self.prophet.predict(horizon=horizon, store_id=store_id)
            sub_preds["prophet"] = prophet_df
            columns.append(prophet_df["predicted_value"].values)

        if "xgb" in active:
            xgb_df = self.xgb.predict(X=X_future, store_id=store_id)
            sub_preds["xgb"] = xgb_df
            columns.append(xgb_df["predicted_value"].values)

        if "lstm" in active:
            lstm_df = self.lstm.predict(X=X_future, store_id=store_id)
            sub_preds["lstm"] = lstm_df
            columns.append(lstm_df["predicted_value"].values)

        X_meta = np.column_stack(columns)
        ensemble_point = np.maximum(ridge.predict(X_meta), 0.0)

        # Weighted CI average
        weights = self._normalised_weights(store_id)
        lower_80 = np.zeros(horizon)
        upper_80 = np.zeros(horizon)
        lower_95 = np.zeros(horizon)
        upper_95 = np.zeros(horizon)

        for model_name, w in weights.items():
            if model_name in sub_preds:
                pdf = sub_preds[model_name]
                lower_80 += w * pdf["lower_80"].values[:horizon]
                upper_80 += w * pdf["upper_80"].values[:horizon]
                lower_95 += w * pdf["lower_95"].values[:horizon]
                upper_95 += w * pdf["upper_95"].values[:horizon]

        # Determine forecast dates
        if "prophet" in sub_preds:
            forecast_dates = sub_preds["prophet"]["forecast_date"].values[:horizon]
        elif "xgb" in sub_preds:
            forecast_dates = sub_preds["xgb"]["forecast_date"].values[:horizon]
        else:
            forecast_dates = pd.date_range(
                start=pd.Timestamp.now().normalize() + pd.Timedelta(days=1),
                periods=horizon,
                freq="D",
            )

        result = pd.DataFrame(
            {
                "forecast_date": forecast_dates,
                "store_id": store_id,
                "predicted_value": ensemble_point,
                "lower_80": lower_80,
                "upper_80": upper_80,
                "lower_95": lower_95,
                "upper_95": upper_95,
                "model_name": "Ensemble",
            }
        )
        return result

    # -- Predict all stores ------------------------------------------------
    def predict_all_stores(
        self,
        horizon: int,
        X_future: pd.DataFrame,
    ) -> pd.DataFrame:
        """Forecast for all fitted stores and concatenate.

        Args:
            horizon: Number of future periods.
            X_future: Future feature DataFrame (must contain ``store_id``).

        Returns:
            Concatenated prediction DataFrame.
        """
        frames: List[pd.DataFrame] = []
        for sid in sorted(self.meta_learners.keys()):
            store_future = (
                X_future.loc[X_future["store_id"] == sid].copy() if "store_id" in X_future.columns else X_future.copy()
            )
            frames.append(self.predict(horizon=horizon, X_future=store_future, store_id=sid))
        return pd.concat(frames, ignore_index=True)

    # -- Weights -----------------------------------------------------------
    def get_weights(self) -> Dict[str, Dict[str, float]]:
        """Return normalised meta-learner weights per store.

        Returns:
            Nested dict ``{store_id: {'prophet': w1, 'xgb': w2}}``.
            Weights sum to approximately 1.0 for each store.
        """
        return {sid: self._normalised_weights(sid) for sid in self.meta_learners}

    def _normalised_weights(self, store_id: str) -> Dict[str, float]:
        """Compute normalised Ridge weights for *store_id*.

        Args:
            store_id: Store identifier.

        Returns:
            Dict mapping model name to its normalised weight.
        """
        ridge = self.meta_learners[store_id]
        active = self.active_models[store_id]
        raw_weights = ridge.coef_

        # Normalise to sum to 1 (using softmax-like absolute normalisation)
        abs_sum = np.sum(np.abs(raw_weights))
        if abs_sum < 1e-12:
            # Equal weights if Ridge produced near-zero coefficients
            normed = np.ones(len(active)) / len(active)
        else:
            normed = np.abs(raw_weights) / abs_sum

        weight_dict: Dict[str, float] = {}
        for i, name in enumerate(active):
            weight_dict[name] = float(normed[i])
        return weight_dict

    # -- Repr --------------------------------------------------------------
    def __repr__(self) -> str:
        stores = list(self.meta_learners.keys())
        return f"EnsembleForecaster(quality_threshold={self.quality_threshold_mape}, " f"stores_fitted={stores})"


# ---------------------------------------------------------------------------
# Inline unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from unittest.mock import MagicMock

    def _make_mock_prophet(
        store_ids: List[str],
        n_val: int,
        mape_values: Optional[Dict[str, float]] = None,
    ) -> MagicMock:
        """Create a mock ProphetForecaster.

        Args:
            store_ids: Store IDs to simulate.
            n_val: Number of validation rows per store.
            mape_values: Optional MAPE values to simulate per store.

        Returns:
            MagicMock mimicking ProphetForecaster.predict().
        """
        mock = MagicMock()

        def predict_side_effect(horizon: int, store_id: str) -> pd.DataFrame:
            rng = np.random.default_rng(hash(store_id) % (2**31))
            vals = rng.uniform(800, 1200, horizon)
            return pd.DataFrame(
                {
                    "forecast_date": pd.date_range("2025-06-01", periods=horizon),
                    "store_id": store_id,
                    "predicted_value": vals,
                    "lower_80": vals * 0.9,
                    "upper_80": vals * 1.1,
                    "lower_95": vals * 0.85,
                    "upper_95": vals * 1.15,
                    "model_name": "Prophet",
                }
            )

        mock.predict = MagicMock(side_effect=predict_side_effect)
        return mock

    def _make_mock_xgb(
        store_ids: List[str],
        n_val: int,
    ) -> MagicMock:
        """Create a mock XGBoostForecaster.

        Args:
            store_ids: Store IDs to simulate.
            n_val: Number of validation rows per store.

        Returns:
            MagicMock mimicking XGBoostForecaster.predict().
        """
        mock = MagicMock()

        def predict_side_effect(X: pd.DataFrame, store_id: str) -> pd.DataFrame:
            n = len(X)
            rng = np.random.default_rng(hash(store_id) % (2**31) + 1)
            vals = rng.uniform(800, 1200, n)
            return pd.DataFrame(
                {
                    "forecast_date": pd.date_range("2025-06-01", periods=n),
                    "store_id": store_id,
                    "predicted_value": vals,
                    "lower_80": vals * 0.9,
                    "upper_80": vals * 1.1,
                    "lower_95": vals * 0.85,
                    "upper_95": vals * 1.15,
                    "model_name": "XGBoost",
                }
            )

        mock.predict = MagicMock(side_effect=predict_side_effect)
        return mock

    def _make_val_df(n: int = 30, store_id: str = "store_0") -> pd.DataFrame:
        """Create synthetic validation DataFrame.

        Args:
            n: Number of rows.
            store_id: Store identifier.

        Returns:
            DataFrame with features and target.
        """
        rng = np.random.default_rng(42)
        return pd.DataFrame(
            {
                "date": pd.date_range("2025-06-01", periods=n),
                "store_id": store_id,
                "total_revenue": rng.uniform(800, 1200, n),
                "lag_1": rng.uniform(700, 1100, n),
                "dow": np.tile(np.arange(7), n // 7 + 1)[:n].astype(float),
                "rolling_7": rng.uniform(850, 1150, n),
            }
        )

    FEATURE_COLS = ["lag_1", "dow", "rolling_7"]

    def test_quality_gate_pass() -> None:
        """Both models pass → ensemble should fit without error."""
        val = _make_val_df()
        prophet_mock = _make_mock_prophet(["store_0"], 30)
        xgb_mock = _make_mock_xgb(["store_0"], 30)
        ens = EnsembleForecaster(prophet_mock, xgb_mock, quality_threshold_mape=100.0)
        ens.fit(val, feature_cols=FEATURE_COLS)
        assert "store_0" in ens.meta_learners
        print("PASS: test_quality_gate_pass")

    def test_quality_gate_fail_one() -> None:
        """One model exceeds threshold → ensemble should still fit."""
        val = _make_val_df()
        prophet_mock = _make_mock_prophet(["store_0"], 30)
        # XGBoost returns garbage predictions → high MAPE
        xgb_mock = MagicMock()
        xgb_mock.predict = MagicMock(
            side_effect=lambda X, store_id: pd.DataFrame(
                {
                    "forecast_date": pd.date_range("2025-06-01", periods=len(X)),
                    "store_id": store_id,
                    "predicted_value": np.zeros(len(X)),
                    "lower_80": np.zeros(len(X)),
                    "upper_80": np.zeros(len(X)),
                    "lower_95": np.zeros(len(X)),
                    "upper_95": np.zeros(len(X)),
                    "model_name": "XGBoost",
                }
            )
        )
        ens = EnsembleForecaster(prophet_mock, xgb_mock, quality_threshold_mape=100.0)
        ens.fit(val, feature_cols=FEATURE_COLS)
        assert "store_0" in ens.meta_learners
        print("PASS: test_quality_gate_fail_one")

    def test_quality_gate_fail_both() -> None:
        """Both models fail → EnsembleQualityError must be raised."""
        val = _make_val_df()
        # Both models return zeros → MAPE = 100%
        bad_prophet = MagicMock()
        bad_prophet.predict = MagicMock(
            side_effect=lambda horizon, store_id: pd.DataFrame(
                {
                    "forecast_date": pd.date_range("2025-06-01", periods=horizon),
                    "store_id": store_id,
                    "predicted_value": np.zeros(horizon),
                    "lower_80": np.zeros(horizon),
                    "upper_80": np.zeros(horizon),
                    "lower_95": np.zeros(horizon),
                    "upper_95": np.zeros(horizon),
                    "model_name": "Prophet",
                }
            )
        )
        bad_xgb = MagicMock()
        bad_xgb.predict = MagicMock(
            side_effect=lambda X, store_id: pd.DataFrame(
                {
                    "forecast_date": pd.date_range("2025-06-01", periods=len(X)),
                    "store_id": store_id,
                    "predicted_value": np.zeros(len(X)),
                    "lower_80": np.zeros(len(X)),
                    "upper_80": np.zeros(len(X)),
                    "lower_95": np.zeros(len(X)),
                    "upper_95": np.zeros(len(X)),
                    "model_name": "XGBoost",
                }
            )
        )
        ens = EnsembleForecaster(bad_prophet, bad_xgb, quality_threshold_mape=25.0)
        try:
            ens.fit(val, feature_cols=FEATURE_COLS)
            assert False, "Should have raised EnsembleQualityError"
        except EnsembleQualityError:
            print("PASS: test_quality_gate_fail_both")

    def test_weight_sum() -> None:
        """Normalised weights should sum to approximately 1."""
        val = _make_val_df()
        prophet_mock = _make_mock_prophet(["store_0"], 30)
        xgb_mock = _make_mock_xgb(["store_0"], 30)
        ens = EnsembleForecaster(prophet_mock, xgb_mock, quality_threshold_mape=100.0)
        ens.fit(val, feature_cols=FEATURE_COLS)
        weights = ens.get_weights()
        total = sum(weights["store_0"].values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ≈ 1.0"
        print("PASS: test_weight_sum")

    def test_prediction_shape() -> None:
        """Ensemble predictions must have the correct number of rows."""
        val = _make_val_df(n=30)
        prophet_mock = _make_mock_prophet(["store_0"], 30)
        xgb_mock = _make_mock_xgb(["store_0"], 30)
        ens = EnsembleForecaster(prophet_mock, xgb_mock, quality_threshold_mape=100.0)
        ens.fit(val, feature_cols=FEATURE_COLS)
        X_future = _make_val_df(n=14)
        preds = ens.predict(horizon=14, X_future=X_future, store_id="store_0")
        assert len(preds) == 14, f"Expected 14 rows, got {len(preds)}"
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
        print("PASS: test_prediction_shape")

    test_quality_gate_pass()
    test_quality_gate_fail_one()
    test_quality_gate_fail_both()
    test_weight_sum()
    test_prediction_shape()
    print("\nAll Ensemble inline tests passed ✓")
    sys.exit(0)
