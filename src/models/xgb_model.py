"""XGBoost gradient-boosted tree forecaster for ADIP.

Uses **Optuna** for Bayesian hyperparameter optimisation and **SHAP** for
post-hoc feature-importance attribution.  One ``XGBRegressor`` is trained
per ``store_id``; the temporal train/validation split is 80/20 by date.

Confidence intervals are computed empirically from residual standard
deviation on the validation fold: ±1.28·σ for 80 % and ±1.96·σ for 95 %.

Typical usage::

    forecaster = XGBoostForecaster(n_optuna_trials=50)
    forecaster.fit(df, feature_cols=['lag_1', 'lag_7', 'dow', 'rolling_7'])
    preds = forecaster.predict(X_test, store_id='downtown')
    importances = forecaster.get_feature_importance('downtown')
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base (standalone mirror of BaseForecaster)
# ---------------------------------------------------------------------------
class BaseForecaster(ABC):
    """Minimal abstract base for all ADIP forecasters."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, **kwargs: Any) -> "BaseForecaster":
        """Train the model."""

    @abstractmethod
    def predict(self, horizon: int, store_id: str, **kwargs: Any) -> pd.DataFrame:
        """Generate forecasts."""


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
# Helper: MAPE
# ---------------------------------------------------------------------------
def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%), ignoring zero actuals.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage.  ``inf`` when all actuals are zero.
    """
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


# ---------------------------------------------------------------------------
# XGBoost Forecaster
# ---------------------------------------------------------------------------
class XGBoostForecaster(BaseForecaster):
    """Gradient-boosted tree forecaster with Optuna tuning and SHAP.

    Attributes:
        n_optuna_trials: Number of Optuna trials per store.
        random_state: Random seed for reproducibility.
        models: Mapping from ``store_id`` to the fitted ``XGBRegressor``.
        best_params: Mapping from ``store_id`` to the best hyper-parameters.
        shap_values: Mapping from ``store_id`` to SHAP value matrix.
        feature_cols: List of feature column names used during training.
        residual_std: Mapping from ``store_id`` to residual σ on validation.
    """

    def __init__(
        self,
        n_optuna_trials: int = 100,
        random_state: int = 42,
    ) -> None:
        """Initialise the XGBoost forecaster.

        Args:
            n_optuna_trials: Number of Bayesian optimisation trials per store.
            random_state: Seed for XGBRegressor and numpy.
        """
        self.n_optuna_trials: int = n_optuna_trials
        self.random_state: int = random_state

        self.models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.shap_values: Dict[str, np.ndarray] = {}
        self.feature_cols: List[str] = []
        self.residual_std: Dict[str, float] = {}
        self._training_features: Dict[str, List[str]] = {}

    # -- Optuna objective --------------------------------------------------
    def _optuna_objective(
        self,
        trial: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Objective function evaluated by Optuna to minimise MAPE.

        Args:
            trial: An Optuna ``Trial`` instance.
            X_train: Training feature matrix.
            y_train: Training target vector.
            X_val: Validation feature matrix.
            y_val: Validation target vector.

        Returns:
            Validation MAPE (lower is better).
        """
        import xgboost as xgb  # noqa: WPS433

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "random_state": self.random_state,
            "objective": "reg:squarederror",
            "verbosity": 0,
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = model.predict(X_val)
        return _safe_mape(y_val, y_pred)

    # -- Fit ---------------------------------------------------------------
    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "total_revenue",
        **kwargs: Any,
    ) -> "XGBoostForecaster":
        """Train an XGBRegressor per store with Optuna hyper-param search.

        The data is split 80/20 chronologically (by ``date`` column).
        After tuning, SHAP values are computed for global interpretability.

        Args:
            df: DataFrame with ``date``, ``store_id``, *feature_cols*, and
                *target_col*.
            feature_cols: Column names to use as input features.
            target_col: Name of the target column.
            **kwargs: Reserved for future use.

        Returns:
            ``self``, enabling method chaining.

        Raises:
            ValueError: If *target_col* or any of *feature_cols* is missing.
        """
        import optuna  # noqa: WPS433
        import xgboost as xgb  # noqa: WPS433

        missing = [c for c in [target_col] + feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {missing}")

        self.feature_cols = list(feature_cols)
        store_ids: List[str] = sorted(df["store_id"].unique().tolist())
        logger.info(
            "Fitting XGBoost for %d store(s) with %d Optuna trials each",
            len(store_ids),
            self.n_optuna_trials,
        )

        _mlflow_log_param("xgb_n_optuna_trials", self.n_optuna_trials)
        _mlflow_log_param("xgb_feature_cols", str(feature_cols))

        for sid in store_ids:
            store_df = (
                df.loc[df["store_id"] == sid]
                .sort_values("date")
                .reset_index(drop=True)
            )
            n = len(store_df)
            split_idx = int(n * 0.8)

            train_slice = store_df.iloc[:split_idx]
            val_slice = store_df.iloc[split_idx:]

            X_train = train_slice[feature_cols].values.astype(np.float32)
            y_train = train_slice[target_col].values.astype(np.float32)
            X_val = val_slice[feature_cols].values.astype(np.float32)
            y_val = val_slice[target_col].values.astype(np.float32)

            logger.info(
                "  [%s] Train=%d  Val=%d", sid, len(X_train), len(X_val)
            )

            # Optuna study – minimise MAPE
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study = optuna.create_study(direction="minimize")
            study.optimize(
                lambda trial: self._optuna_objective(
                    trial, X_train, y_train, X_val, y_val
                ),
                n_trials=self.n_optuna_trials,
                show_progress_bar=False,
            )

            best = study.best_params
            best["random_state"] = self.random_state
            best["objective"] = "reg:squarederror"
            best["verbosity"] = 0
            self.best_params[sid] = best

            logger.info("  [%s] Best params: %s", sid, best)
            _mlflow_log_param(f"xgb_{sid}_best_params", str(best))
            _mlflow_log_metric(f"xgb_{sid}_best_mape", study.best_value)

            # Final model trained on full training set
            final_model = xgb.XGBRegressor(**best)
            final_model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            self.models[sid] = final_model
            self._training_features[sid] = list(feature_cols)

            # Residual std for empirical CIs
            val_preds = final_model.predict(X_val)
            residuals = y_val - val_preds
            self.residual_std[sid] = float(np.std(residuals))
            _mlflow_log_metric(
                f"xgb_{sid}_residual_std", self.residual_std[sid]
            )

            # SHAP values
            try:
                import shap  # noqa: WPS433

                explainer = shap.TreeExplainer(final_model)
                sv = explainer.shap_values(X_val)
                self.shap_values[sid] = sv
                logger.info(
                    "  [%s] SHAP values computed: shape=%s", sid, sv.shape
                )
            except ImportError:
                logger.warning(
                    "  [%s] shap not installed — skipping importance", sid
                )
            except Exception:
                logger.warning(
                    "  [%s] SHAP computation failed", sid, exc_info=True
                )

        return self

    # -- Predict -----------------------------------------------------------
    def predict(
        self,
        X: pd.DataFrame,
        store_id: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Generate predictions with empirical confidence intervals.

        Intervals are derived from the validation-set residual σ:

        * 80 % CI → ±1.28 σ
        * 95 % CI → ±1.96 σ

        Predicted values and lower bounds are clipped to ≥ 0.

        Args:
            X: Future feature DataFrame (must contain the same
                *feature_cols* used in :meth:`fit`).
            store_id: Store to predict for.

        Returns:
            DataFrame with the ADIP standard forecast schema.

        Raises:
            KeyError: If *store_id* was not fitted.
        """
        if store_id not in self.models:
            raise KeyError(
                f"No fitted model for store_id='{store_id}'. "
                f"Available: {list(self.models.keys())}"
            )

        model = self.models[store_id]
        feat_cols = self._training_features.get(store_id, self.feature_cols)

        X_arr = X[feat_cols].values.astype(np.float32)
        point = model.predict(X_arr)
        sigma = self.residual_std.get(store_id, 0.0)

        result = pd.DataFrame(
            {
                "forecast_date": (
                    X["date"].values if "date" in X.columns
                    else pd.RangeIndex(len(point))
                ),
                "store_id": store_id,
                "predicted_value": np.maximum(point, 0.0),
                "lower_80": np.maximum(point - 1.28 * sigma, 0.0),
                "upper_80": point + 1.28 * sigma,
                "lower_95": np.maximum(point - 1.96 * sigma, 0.0),
                "upper_95": point + 1.96 * sigma,
                "model_name": "XGBoost",
            }
        )
        return result

    # -- Feature importance ------------------------------------------------
    def get_feature_importance(self, store_id: str) -> pd.DataFrame:
        """Return SHAP-based feature importances for a store.

        Args:
            store_id: Store whose model to inspect.

        Returns:
            DataFrame with ``feature_name``, ``mean_shap_value``, ``rank``
            sorted by descending importance.

        Raises:
            KeyError: If *store_id* has no SHAP values (not fitted or SHAP
                computation failed).
        """
        if store_id not in self.shap_values:
            raise KeyError(
                f"No SHAP values for store_id='{store_id}'. "
                f"Available: {list(self.shap_values.keys())}"
            )

        sv = self.shap_values[store_id]
        mean_abs = np.mean(np.abs(sv), axis=0)
        feat_cols = self._training_features.get(store_id, self.feature_cols)

        importance_df = pd.DataFrame(
            {"feature_name": feat_cols, "mean_shap_value": mean_abs}
        )
        importance_df = importance_df.sort_values(
            "mean_shap_value", ascending=False
        ).reset_index(drop=True)
        importance_df["rank"] = importance_df.index + 1
        return importance_df

    # -- Repr --------------------------------------------------------------
    def __repr__(self) -> str:
        fitted = list(self.models.keys())
        return (
            f"XGBoostForecaster(n_optuna_trials={self.n_optuna_trials}, "
            f"stores_fitted={fitted})"
        )


# ---------------------------------------------------------------------------
# Inline unit tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    def _make_synthetic(
        n_days: int = 120,
        n_stores: int = 2,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate synthetic daily data with engineered features.

        Args:
            n_days: Number of days.
            n_stores: Number of stores.
            seed: Random seed.

        Returns:
            DataFrame ready for XGBoost training.
        """
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_stores):
            dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
            dow = dates.dayofweek.values.astype(float)
            trend = np.linspace(800, 1200, n_days)
            weekly = 150 * np.sin(2 * np.pi * dow / 7)
            noise = rng.normal(0, 40, n_days)
            revenue = trend + weekly + noise + s * 200

            lag_1 = np.concatenate([[revenue[0]], revenue[:-1]])
            lag_7 = np.concatenate([revenue[:7], revenue[:-7]])
            rolling_7 = pd.Series(revenue).rolling(7, min_periods=1).mean().values

            for i, d in enumerate(dates):
                rows.append(
                    {
                        "date": d,
                        "store_id": f"store_{s}",
                        "total_revenue": revenue[i],
                        "lag_1": lag_1[i],
                        "lag_7": lag_7[i],
                        "dow": dow[i],
                        "rolling_7": rolling_7[i],
                    }
                )
        return pd.DataFrame(rows)

    FEATURE_COLS = ["lag_1", "lag_7", "dow", "rolling_7"]

    def test_fit_completes() -> None:
        """XGBoost should fit without error."""
        df = _make_synthetic(n_days=100, n_stores=1)
        fc = XGBoostForecaster(n_optuna_trials=5, random_state=42)
        fc.fit(df, feature_cols=FEATURE_COLS)
        assert "store_0" in fc.models
        print("PASS: test_fit_completes")

    def test_predict_schema() -> None:
        """Output must follow ADIP prediction schema."""
        df = _make_synthetic(n_days=100, n_stores=1)
        fc = XGBoostForecaster(n_optuna_trials=5, random_state=42)
        fc.fit(df, feature_cols=FEATURE_COLS)
        X_future = df.tail(10).copy()
        preds = fc.predict(X_future, store_id="store_0")
        expected = {
            "forecast_date", "store_id", "predicted_value",
            "lower_80", "upper_80", "lower_95", "upper_95", "model_name",
        }
        assert set(preds.columns) == expected
        print("PASS: test_predict_schema")

    def test_non_negative() -> None:
        """Predicted values must be ≥ 0."""
        df = _make_synthetic(n_days=100, n_stores=1)
        fc = XGBoostForecaster(n_optuna_trials=5, random_state=42)
        fc.fit(df, feature_cols=FEATURE_COLS)
        preds = fc.predict(df.tail(10), store_id="store_0")
        assert (preds["predicted_value"] >= 0).all()
        assert (preds["lower_80"] >= 0).all()
        print("PASS: test_non_negative")

    def test_ci_ordering() -> None:
        """CI bounds must satisfy lower_95 ≤ lower_80 ≤ upper_80 ≤ upper_95."""
        df = _make_synthetic(n_days=100, n_stores=1)
        fc = XGBoostForecaster(n_optuna_trials=5, random_state=42)
        fc.fit(df, feature_cols=FEATURE_COLS)
        preds = fc.predict(df.tail(10), store_id="store_0")
        assert (preds["lower_95"] <= preds["lower_80"]).all()
        assert (preds["lower_80"] <= preds["upper_80"]).all()
        assert (preds["upper_80"] <= preds["upper_95"]).all()
        print("PASS: test_ci_ordering")

    def test_feature_importance() -> None:
        """Feature importance should return all features, ranked."""
        df = _make_synthetic(n_days=100, n_stores=1)
        fc = XGBoostForecaster(n_optuna_trials=5, random_state=42)
        fc.fit(df, feature_cols=FEATURE_COLS)
        imp = fc.get_feature_importance("store_0")
        assert len(imp) == len(FEATURE_COLS)
        assert list(imp["rank"]) == [1, 2, 3, 4]
        print("PASS: test_feature_importance")

    test_fit_completes()
    test_predict_schema()
    test_non_negative()
    test_ci_ordering()
    test_feature_importance()
    print("\nAll XGBoost inline tests passed ✓")
    sys.exit(0)
