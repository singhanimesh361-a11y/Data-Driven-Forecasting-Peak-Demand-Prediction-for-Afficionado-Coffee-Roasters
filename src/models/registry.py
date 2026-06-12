"""MLflow Model Registry and training pipeline for ADIP."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import mlflow
import pandas as pd

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Wraps MLflow functionality for the ADIP platform."""

    def __init__(self, tracking_uri: str = "sqlite:///mlruns.db", experiment_name: str = "adip-forecasting"):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info(
            f"Initialized ModelRegistry with tracking URI: {self.tracking_uri}, experiment: {self.experiment_name}"
        )

    def start_run(self, model_name: str, store_id: int) -> mlflow.ActiveRun:
        """Creates a run tagged with model_name, store_id, and run timestamp."""
        run = mlflow.start_run()
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("store_id", str(store_id))
        mlflow.set_tag("run_timestamp", datetime.now().isoformat())
        return run

    def log_training_run(
        self,
        run: mlflow.ActiveRun,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model_obj: Any,
        artifact_paths: Optional[List[str]] = None,
    ) -> str:
        """Logs params, metrics, pickled model, and any artifact paths."""
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        import os
        import pickle

        model_path = "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
        mlflow.log_artifact(model_path, "model")
        os.remove(model_path)

        if artifact_paths:
            for path in artifact_paths:
                mlflow.log_artifact(path)

        run_id = run.info.run_id
        logger.info(f"Logged run {run_id} successfully.")
        return run_id

    def load_best_model(
        self, model_name: str, store_id: int, metric: str = "mape", lower_is_better: bool = True
    ) -> Any:
        """Query MLflow runs filtered by model_name and store_id tags, return best model."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {self.experiment_name} not found")

        query = f"tags.model_name = '{model_name}' and tags.store_id = '{store_id}'"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=query,
            order_by=[f"metrics.{metric} {'ASC' if lower_is_better else 'DESC'}"],
        )
        if not runs:
            raise ValueError(f"No runs found for model {model_name} and store {store_id}")

        best_run = runs[0]
        local_path = client.download_artifacts(best_run.info.run_id, "model/model.pkl")
        import pickle

        with open(local_path, "rb") as f:
            model = pickle.load(f)
        return model

    def list_experiments(self) -> pd.DataFrame:
        """Return DataFrame with run details."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return pd.DataFrame()
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        data = []
        for r in runs:
            data.append(
                {
                    "run_id": r.info.run_id,
                    "model_name": r.data.tags.get("model_name"),
                    "store_id": r.data.tags.get("store_id"),
                    "mape": r.data.metrics.get("mape"),
                    "rmse": r.data.metrics.get("rmse"),
                    "mae": r.data.metrics.get("mae"),
                    "run_date": r.data.tags.get("run_timestamp"),
                    "artifact_uri": r.info.artifact_uri,
                }
            )
        return pd.DataFrame(data)

    def get_latest_run_id(self, model_name: str, store_id: int) -> str:
        """Return run_id of the most recent run for given model + store."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            raise ValueError(f"Experiment {self.experiment_name} not found")
        query = f"tags.model_name = '{model_name}' and tags.store_id = '{store_id}'"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id], filter_string=query, order_by=["start_time DESC"]
        )
        if not runs:
            raise ValueError(f"No runs found for {model_name} and {store_id}")
        return runs[0].info.run_id


def train_all_models(db_path: str, registry: ModelRegistry) -> None:
    """Pipeline to train all models, evaluate, and log to MLflow.

    Args:
        db_path: Path to SQLite DB.
        registry: ModelRegistry instance.
    """
    import sqlite3

    from src.evaluation.metrics import ModelEvaluator
    from src.features.feature_pipeline import FeaturePipeline

    logger.info("Loading daily_store from SQLite")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * FROM daily_store", conn)

    logger.info("Running FeaturePipeline")
    fp = FeaturePipeline(target_col="total_revenue", freq="D")
    df_features = fp.fit_transform(df)

    evaluator = ModelEvaluator()
    results = []

    models_to_train = ["Naive", "SeasonalNaive", "MovingAverage", "SARIMA", "Prophet", "XGBoost", "LSTM", "Ensemble"]
    logger.info(f"Models to train: {models_to_train}")

    from src.models.baseline import MovingAverageForecaster, NaiveForecaster, SeasonalNaiveForecaster
    from src.models.ensemble import EnsembleForecaster
    from src.models.lstm_model import LSTMForecaster
    from src.models.prophet_model import ProphetForecaster
    from src.models.sarima_model import SARIMAForecaster
    from src.models.xgb_model import XGBoostForecaster

    df_features["date"] = pd.to_datetime(df_features["date"])
    max_date = df_features["date"].max()
    split_date = max_date - pd.Timedelta(days=30)
    train_df = df_features[df_features["date"] <= split_date].copy()
    test_df = df_features[df_features["date"] > split_date].copy()

    store_ids = df_features["store_id"].unique()

    for store_id in store_ids:
        store_train = train_df[train_df["store_id"] == store_id].copy()
        store_test = test_df[test_df["store_id"] == store_id].copy()

        trained_models = {}
        for model_name in models_to_train:
            try:
                logger.info(f"Training {model_name} for store {store_id}")
                if model_name == "Naive":
                    model = NaiveForecaster().fit(store_train, target_col="total_revenue")
                elif model_name == "SeasonalNaive":
                    model = SeasonalNaiveForecaster().fit(store_train, target_col="total_revenue")
                elif model_name == "MovingAverage":
                    model = MovingAverageForecaster(window=7).fit(store_train, target_col="total_revenue")
                elif model_name == "SARIMA":
                    model = SARIMAForecaster().fit(store_train)
                elif model_name == "Prophet":
                    model = ProphetForecaster(freq="D").fit(store_train)
                elif model_name == "XGBoost":
                    model = XGBoostForecaster(n_optuna_trials=10).fit(store_train, feature_cols=fp.get_feature_names())
                elif model_name == "LSTM":
                    model = LSTMForecaster(n_optuna_trials=3, epochs=10).fit(
                        store_train, feature_cols=fp.get_feature_names()
                    )
                elif model_name == "Ensemble":
                    if "Prophet" in trained_models and "XGBoost" in trained_models:
                        model = EnsembleForecaster(
                            prophet=trained_models["Prophet"],
                            xgb=trained_models["XGBoost"],
                            lstm=trained_models.get("LSTM"),
                        )
                        model.fit(store_test, feature_cols=fp.get_feature_names())
                    else:
                        logger.warning(f"Skipping Ensemble for store {store_id} - missing constituents")
                        continue

                trained_models[model_name] = model

                horizon = len(store_test)
                if model_name in ["XGBoost", "LSTM"]:
                    preds = model.predict(store_test, store_id=store_id)
                elif model_name == "Ensemble":
                    preds = model.predict(horizon=horizon, X_future=store_test, store_id=store_id)
                elif model_name in ["Prophet", "SARIMA"]:
                    preds = model.predict(horizon=horizon, store_id=store_id)
                else:
                    preds = model.predict(horizon=horizon, store_ids=[store_id])

                metrics = evaluator.evaluate(
                    y_true=store_test["total_revenue"].values,
                    y_pred=preds["predicted_value"].values,
                    lower_95=preds.get("lower_95", pd.Series(dtype=float)).values if "lower_95" in preds else None,
                    upper_95=preds.get("upper_95", pd.Series(dtype=float)).values if "upper_95" in preds else None,
                )
                metrics["model"] = model_name
                metrics["store_id"] = store_id

                results.append(metrics)

                with registry.start_run(model_name, store_id) as run:
                    registry.log_training_run(
                        run=run,
                        params={"model_type": model_name},
                        metrics={"mape": metrics["mape"], "rmse": metrics["rmse"], "mae": metrics["mae"]},
                        model_obj=model,
                    )

            except Exception as e:
                logger.error(f"Failed to train {model_name} for store {store_id}: {e}")
                continue

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        print("\n--- MODEL LEADERBOARD ---")
        leaderboard = df_results.sort_values("mape")
        print(leaderboard[["model", "store_id", "mape", "rmse", "peak_error_rate"]].to_string(index=False))
