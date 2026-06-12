"""
Afficionado Demand Intelligence Platform — Forecast Engine
===========================================================
Production forecast generation service with scheduling, retry logic,
structured logging, and data-freshness validation.

Usage:
    python -m src.serving.forecast_engine --mode=daemon   # Nightly at 02:00
    python -m src.serving.forecast_engine --mode=once      # Single run
    python -m src.serving.forecast_engine --mode=healthcheck
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """Emit structured JSON log lines for production log aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = traceback.format_exception(*record.exc_info)
        # Merge extra fields
        for key in ("store", "category", "horizon", "duration_s", "rows", "metric"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def _get_logger(name: str = "adip.forecast_engine") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = _get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORES: List[str] = ["Store_1", "Store_2", "Store_3"]
CATEGORIES: List[str] = [
    "Coffee", "Tea", "Bakery", "Drinking Chocolate", "Coffee beans",
    "Branded", "Loose Tea", "Flavours", "Packaged Chocolate",
]
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # seconds
DATA_FRESHNESS_HOURS = 25  # data older than this → stale


# ---------------------------------------------------------------------------
# Retry decorator with exponential backoff
# ---------------------------------------------------------------------------

def retry_with_backoff(max_retries: int = MAX_RETRIES, backoff_base: int = RETRY_BACKOFF_BASE):
    """Decorator: retry a function with exponential back-off on exception."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    wait = backoff_base ** attempt
                    logger.warning(
                        "Attempt %d/%d for %s failed — retrying in %ds",
                        attempt, max_retries, func.__name__, wait,
                    )
                    time.sleep(wait)
            logger.error("All %d attempts for %s exhausted", max_retries, func.__name__)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# ForecastEngine
# ---------------------------------------------------------------------------

class ForecastEngine:
    """
    Core forecast generation engine.

    Parameters
    ----------
    db_path : str | Path
        Path to the processed transaction database (Parquet / SQLite).
    forecast_dir : str | Path
        Output directory for forecast artefacts.
    registry : str | Path | None
        Path or URI to the MLflow model registry / local model store.
    """

    def __init__(
        self,
        db_path: str | Path = "data/processed/coffee_shop_sales_cleaned.parquet",
        forecast_dir: str | Path = "data/forecast_store",
        registry: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.forecast_dir = Path(forecast_dir)
        self.forecast_dir.mkdir(parents=True, exist_ok=True)
        self.registry = Path(registry) if registry else Path("models/registry")
        self._run_id: str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        logger.info(
            "ForecastEngine initialised",
            extra={"store": "all", "category": "all"},
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """Load and validate the transaction dataset."""
        if self.db_path.suffix == ".parquet":
            df = pd.read_parquet(self.db_path)
        elif self.db_path.suffix in (".csv", ".gz"):
            df = pd.read_csv(self.db_path, parse_dates=["transaction_date"])
        else:
            raise ValueError(f"Unsupported data format: {self.db_path.suffix}")

        required_cols = {"transaction_date", "store_id", "product_category", "transaction_qty", "unit_price"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        logger.info("Loaded %d rows from %s", len(df), self.db_path, extra={"rows": len(df)})
        return df

    def _load_best_model(self, store: str, category: str):
        """
        Load the best-performing model for a (store, category) pair.

        Looks in the local model registry first, then falls back to a Prophet
        baseline.
        """
        model_path = self.registry / f"{store}_{category}_best.pkl"
        if model_path.exists():
            import pickle
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info("Loaded model from %s", model_path)
            return model

        # Fallback: train a quick Prophet model
        from prophet import Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        logger.info(
            "No cached model for %s/%s — will train Prophet baseline",
            store, category,
        )
        return model

    # ------------------------------------------------------------------
    # Forecast generation
    # ------------------------------------------------------------------

    @retry_with_backoff()
    def generate_daily_forecasts(self, horizon: int = 30) -> pd.DataFrame:
        """
        Generate daily demand forecasts for all stores × categories.

        Parameters
        ----------
        horizon : int
            Number of days to forecast ahead.

        Returns
        -------
        pd.DataFrame
            Combined forecast dataframe written to Parquet.
        """
        from prophet import Prophet

        t0 = time.time()
        df = self._load_data()
        all_forecasts: List[pd.DataFrame] = []

        for store in STORES:
            for category in CATEGORIES:
                store_cat_df = df[
                    (df["store_id"] == store) & (df["product_category"] == category)
                ].copy()

                if store_cat_df.empty:
                    logger.warning("No data for %s / %s — skipping", store, category)
                    continue

                # Aggregate daily
                daily = (
                    store_cat_df
                    .groupby(pd.Grouper(key="transaction_date", freq="D"))
                    .agg(y=("transaction_qty", "sum"))
                    .reset_index()
                    .rename(columns={"transaction_date": "ds"})
                )

                # Load or create model
                model = self._load_best_model(store, category)
                if not hasattr(model, "history"):
                    # Needs fitting
                    model.fit(daily)

                future = model.make_future_dataframe(periods=horizon, freq="D")
                forecast = model.predict(future)

                result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon).copy()
                result["store_id"] = store
                result["product_category"] = category
                result["generated_at"] = datetime.utcnow().isoformat()
                result["run_id"] = self._run_id
                all_forecasts.append(result)

                logger.info(
                    "Forecast generated: %s / %s — %d days",
                    store, category, horizon,
                    extra={"store": store, "category": category, "horizon": horizon},
                )

        if not all_forecasts:
            raise RuntimeError("No forecasts were generated — check data availability")

        combined = pd.concat(all_forecasts, ignore_index=True)

        # Write output
        out_path = self.forecast_dir / f"daily_forecast_{self._run_id}.parquet"
        combined.to_parquet(out_path, index=False)

        # Also write a "latest" symlink / copy
        latest_path = self.forecast_dir / "daily_forecast_latest.parquet"
        combined.to_parquet(latest_path, index=False)

        elapsed = time.time() - t0
        logger.info(
            "Daily forecasts complete — %d rows in %.1fs",
            len(combined), elapsed,
            extra={"rows": len(combined), "duration_s": round(elapsed, 2)},
        )
        return combined

    @retry_with_backoff()
    def generate_hourly_forecasts(self, horizon_hours: int = 72) -> pd.DataFrame:
        """
        Generate hourly demand forecasts for operational scheduling.

        Parameters
        ----------
        horizon_hours : int
            Number of hours to forecast ahead (default: 72 = 3 days).

        Returns
        -------
        pd.DataFrame
            Combined hourly forecast dataframe.
        """
        from prophet import Prophet

        t0 = time.time()
        df = self._load_data()

        # Need an hourly timestamp
        if "transaction_time" in df.columns:
            df["ds"] = pd.to_datetime(
                df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str)
            )
        elif "transaction_datetime" in df.columns:
            df["ds"] = pd.to_datetime(df["transaction_datetime"])
        else:
            df["ds"] = pd.to_datetime(df["transaction_date"])

        all_forecasts: List[pd.DataFrame] = []

        for store in STORES:
            store_df = df[df["store_id"] == store].copy()
            if store_df.empty:
                continue

            hourly = (
                store_df
                .set_index("ds")
                .resample("h")
                .agg(y=("transaction_qty", "sum"))
                .reset_index()
            )
            hourly = hourly[hourly["y"] > 0]

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.1,
            )
            model.fit(hourly)

            future = model.make_future_dataframe(periods=horizon_hours, freq="h")
            forecast = model.predict(future)

            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(horizon_hours).copy()
            result["store_id"] = store
            result["generated_at"] = datetime.utcnow().isoformat()
            result["run_id"] = self._run_id
            all_forecasts.append(result)

            logger.info(
                "Hourly forecast generated: %s — %d hours",
                store, horizon_hours,
                extra={"store": store, "horizon": horizon_hours},
            )

        combined = pd.concat(all_forecasts, ignore_index=True) if all_forecasts else pd.DataFrame()

        out_path = self.forecast_dir / f"hourly_forecast_{self._run_id}.parquet"
        combined.to_parquet(out_path, index=False)
        latest_path = self.forecast_dir / "hourly_forecast_latest.parquet"
        combined.to_parquet(latest_path, index=False)

        elapsed = time.time() - t0
        logger.info(
            "Hourly forecasts complete — %d rows in %.1fs",
            len(combined), elapsed,
            extra={"rows": len(combined), "duration_s": round(elapsed, 2)},
        )
        return combined

    # ------------------------------------------------------------------
    # Data freshness
    # ------------------------------------------------------------------

    def check_data_freshness(self) -> bool:
        """
        Return True if the latest forecast file is fresh (< DATA_FRESHNESS_HOURS old).
        """
        latest_path = self.forecast_dir / "daily_forecast_latest.parquet"
        if not latest_path.exists():
            logger.warning("No latest forecast file found — data is STALE")
            return False

        mtime = datetime.utcfromtimestamp(latest_path.stat().st_mtime)
        age_hours = (datetime.utcnow() - mtime).total_seconds() / 3600

        is_fresh = age_hours < DATA_FRESHNESS_HOURS
        status = "FRESH" if is_fresh else "STALE"
        logger.info(
            "Forecast freshness: %s (age=%.1f hours, threshold=%d hours)",
            status, age_hours, DATA_FRESHNESS_HOURS,
        )
        return is_fresh

    # ------------------------------------------------------------------
    # Nightly scheduled refresh
    # ------------------------------------------------------------------

    def run_scheduled_refresh(self) -> Dict[str, Any]:
        """
        Execute the full nightly refresh pipeline:
        1. Check data freshness of source data
        2. Generate daily forecasts (30-day horizon)
        3. Generate hourly forecasts (72-hour horizon)
        4. Log summary metrics
        """
        t0 = time.time()
        run_report: Dict[str, Any] = {
            "run_id": self._run_id,
            "started_at": datetime.utcnow().isoformat() + "Z",
            "status": "running",
            "steps": {},
        }

        try:
            # Step 1: Source data check
            logger.info("Step 1/4 — Checking source data freshness")
            source_fresh = self.db_path.exists()
            run_report["steps"]["data_check"] = {"source_exists": source_fresh}
            if not source_fresh:
                raise FileNotFoundError(f"Source data not found: {self.db_path}")

            # Step 2: Daily forecasts
            logger.info("Step 2/4 — Generating daily forecasts")
            daily_df = self.generate_daily_forecasts(horizon=30)
            run_report["steps"]["daily_forecast"] = {
                "rows": len(daily_df),
                "stores": daily_df["store_id"].nunique(),
                "categories": daily_df["product_category"].nunique(),
            }

            # Step 3: Hourly forecasts
            logger.info("Step 3/4 — Generating hourly forecasts")
            hourly_df = self.generate_hourly_forecasts(horizon_hours=72)
            run_report["steps"]["hourly_forecast"] = {"rows": len(hourly_df)}

            # Step 4: Summary
            logger.info("Step 4/4 — Writing run report")
            run_report["status"] = "success"

        except Exception as exc:
            run_report["status"] = "failed"
            run_report["error"] = str(exc)
            logger.error("Scheduled refresh FAILED: %s", exc, exc_info=True)
            raise

        finally:
            elapsed = time.time() - t0
            run_report["completed_at"] = datetime.utcnow().isoformat() + "Z"
            run_report["duration_s"] = round(elapsed, 2)

            report_path = self.forecast_dir / f"run_report_{self._run_id}.json"
            with open(report_path, "w") as f:
                json.dump(run_report, f, indent=2)

            logger.info(
                "Scheduled refresh %s in %.1fs",
                run_report["status"].upper(), elapsed,
                extra={"duration_s": run_report["duration_s"]},
            )

        return run_report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _daemon_mode(engine: ForecastEngine) -> None:
    """Run forecast engine as a persistent daemon with APScheduler."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    hour = int(os.environ.get("SCHEDULE_HOUR", "2"))
    minute = int(os.environ.get("SCHEDULE_MINUTE", "0"))

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        engine.run_scheduled_refresh,
        trigger=CronTrigger(hour=hour, minute=minute),
        id="nightly_forecast_refresh",
        name="Nightly Forecast Refresh",
        misfire_grace_time=3600,
        max_instances=1,
        replace_existing=True,
    )

    logger.info("Daemon started — scheduled at %02d:%02d UTC daily", hour, minute)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Daemon shutting down gracefully")
        scheduler.shutdown(wait=True)


def _healthcheck_mode(engine: ForecastEngine) -> None:
    """Check forecast freshness and exit with appropriate code."""
    is_fresh = engine.check_data_freshness()
    if is_fresh:
        print("OK — forecasts are fresh")
        sys.exit(0)
    else:
        print("STALE — forecasts need refresh")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADIP Forecast Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["daemon", "once", "healthcheck"],
        default="once",
        help="Execution mode",
    )
    parser.add_argument("--db-path", default="data/processed/coffee_shop_sales_cleaned.parquet")
    parser.add_argument("--forecast-dir", default="data/forecast_store")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--horizon", type=int, default=30, help="Daily forecast horizon in days")
    args = parser.parse_args()

    engine = ForecastEngine(
        db_path=args.db_path,
        forecast_dir=args.forecast_dir,
        registry=args.registry,
    )

    if args.mode == "daemon":
        _daemon_mode(engine)
    elif args.mode == "healthcheck":
        _healthcheck_mode(engine)
    else:
        engine.run_scheduled_refresh()


if __name__ == "__main__":
    main()
