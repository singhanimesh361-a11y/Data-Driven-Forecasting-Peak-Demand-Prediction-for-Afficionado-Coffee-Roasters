"""
Afficionado Demand Intelligence Platform — Health API
======================================================
FastAPI service exposing health, drift, and metrics endpoints for
external monitoring systems (Prometheus, UptimeRobot, etc.).

Endpoints
---------
GET /health          → 200 if fresh, 503 if stale
GET /health/drift    → Drift status per store × category
GET /health/metrics  → 7-day rolling MAPE per model per store

Usage:
    uvicorn src.serving.health_api:app --host 0.0.0.0 --port 8502
    python -m src.serving.health_api
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from src.serving.monitoring import ForecastMonitor

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ADIP Health API",
    description="Health and monitoring endpoints for the Afficionado Demand Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow dashboard and monitoring tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Shared monitor instance
_monitor = ForecastMonitor(
    forecast_dir=os.environ.get("FORECAST_DIR", "data/forecast_store"),
    metrics_path=os.environ.get("METRICS_PATH", "data/processed/evaluation_metrics.csv"),
    training_metrics_path=os.environ.get("TRAINING_METRICS_PATH", "data/processed/training_metrics.csv"),
)


# ---------------------------------------------------------------------------
# GET /health — primary liveness + freshness check
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    summary="System Health Check",
    description="Returns 200 if forecasts are fresh, 503 if stale or unavailable.",
    responses={
        200: {"description": "System healthy — forecasts are fresh"},
        503: {"description": "System unhealthy — forecasts are stale or missing"},
    },
)
async def health_check(response: Response) -> Dict[str, Any]:
    """
    Primary health endpoint.

    Used by load balancers, container orchestrators, and uptime monitors.
    Returns HTTP 200 when forecasts are fresh, HTTP 503 when stale.
    """
    staleness = _monitor.check_data_staleness()
    availability = _monitor.check_forecast_availability()

    is_healthy = not staleness["is_stale"] and availability.get("available", False)

    result = {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "checks": {
            "forecast_freshness": {
                "status": "pass" if not staleness["is_stale"] else "fail",
                "age_hours": staleness.get("age_hours"),
                "threshold_hours": staleness["threshold_hours"],
            },
            "forecast_availability": {
                "status": "pass" if availability.get("available") else "fail",
                "daily_rows": availability["daily_rows"],
                "hourly_rows": availability["hourly_rows"],
            },
        },
        "version": "1.0.0",
    }

    if not is_healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return result


# ---------------------------------------------------------------------------
# GET /health/drift — model drift status
# ---------------------------------------------------------------------------


@app.get(
    "/health/drift",
    summary="Model Drift Status",
    description="Returns drift analysis for each store × category pair.",
)
async def drift_status() -> Dict[str, Any]:
    """
    Check model drift across all store × category pairs.

    Drift is flagged when recent MAPE exceeds training MAPE by more than
    10 percentage points.
    """
    drift_results = _monitor.check_model_drift()

    # Group by store for cleaner output
    by_store: Dict[str, List[Dict[str, Any]]] = {}
    for result in drift_results:
        store = result.store
        if store not in by_store:
            by_store[store] = []
        by_store[store].append(result.to_dict())

    drifted_count = sum(1 for r in drift_results if r.is_drifted)

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_models": len(drift_results),
            "drifted_models": drifted_count,
            "drift_threshold_pp": 10.0,
            "overall_status": "drift_detected" if drifted_count > 0 else "stable",
        },
        "stores": by_store,
    }


# ---------------------------------------------------------------------------
# GET /health/metrics — 7-day rolling MAPE
# ---------------------------------------------------------------------------


@app.get(
    "/health/metrics",
    summary="Forecast Accuracy Metrics",
    description="Returns 7-day rolling MAPE per model per store.",
)
async def forecast_metrics() -> Dict[str, Any]:
    """
    Compute and return 7-day rolling MAPE for each model and store.

    Reads the latest forecast file and, if actuals are available,
    computes prediction accuracy.
    """
    forecast_dir = Path(os.environ.get("FORECAST_DIR", "data/forecast_store"))
    actuals_path = Path(os.environ.get("ACTUALS_PATH", "data/processed/coffee_shop_sales_cleaned.parquet"))

    metrics: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "window": "7_day_rolling",
        "stores": {},
    }

    # Load latest forecast
    latest_forecast_path = forecast_dir / "daily_forecast_latest.parquet"
    if not latest_forecast_path.exists():
        return {
            **metrics,
            "error": "No forecast file available",
        }

    try:
        forecast_df = pd.read_parquet(latest_forecast_path)
    except Exception as exc:
        return {
            **metrics,
            "error": f"Failed to read forecast file: {exc}",
        }

    # Load actuals if available
    actuals_df = None
    if actuals_path.exists():
        try:
            if actuals_path.suffix == ".parquet":
                actuals_df = pd.read_parquet(actuals_path)
            else:
                actuals_df = pd.read_csv(actuals_path, parse_dates=["transaction_date"])
        except Exception:
            actuals_df = None

    stores = forecast_df["store_id"].unique() if "store_id" in forecast_df.columns else []

    for store in stores:
        store_forecasts = forecast_df[forecast_df["store_id"] == store]
        categories = (
            store_forecasts["product_category"].unique() if "product_category" in store_forecasts.columns else []
        )

        store_metrics: Dict[str, Any] = {}

        for category in categories:
            cat_forecast = store_forecasts[store_forecasts["product_category"] == category]

            entry: Dict[str, Any] = {
                "forecast_rows": len(cat_forecast),
                "forecast_mean": (
                    round(float(cat_forecast["yhat"].mean()), 2) if "yhat" in cat_forecast.columns else None
                ),
            }

            # Compute MAPE if actuals are available
            if actuals_df is not None and "transaction_date" in actuals_df.columns:
                try:
                    store_actuals = actuals_df[
                        (actuals_df["store_id"] == store) & (actuals_df["product_category"] == category)
                    ]
                    if not store_actuals.empty and "yhat" in cat_forecast.columns:
                        # Aggregate actuals daily
                        daily_actuals = (
                            store_actuals.groupby(pd.Grouper(key="transaction_date", freq="D"))
                            .agg(actual=("transaction_qty", "sum"))
                            .reset_index()
                            .rename(columns={"transaction_date": "ds"})
                        )

                        # Merge forecast with actuals
                        cat_forecast_copy = cat_forecast.copy()
                        cat_forecast_copy["ds"] = pd.to_datetime(cat_forecast_copy["ds"])
                        merged = pd.merge(
                            cat_forecast_copy[["ds", "yhat"]],
                            daily_actuals,
                            on="ds",
                            how="inner",
                        )

                        if len(merged) > 0:
                            # 7-day window
                            merged = merged.sort_values("ds").tail(7)
                            mape = float(np.mean(np.abs((merged["actual"] - merged["yhat"]) / merged["actual"])) * 100)
                            entry["mape_7d"] = round(mape, 2)
                            entry["comparison_days"] = len(merged)
                except Exception:
                    pass  # gracefully skip if merge fails

            store_metrics[category] = entry

        metrics["stores"][store] = store_metrics

    return metrics


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event():
    """Log startup and run initial health check."""
    import logging

    logger = logging.getLogger("adip.health_api")
    logger.info("ADIP Health API starting on port %s", os.environ.get("PORT", "8502"))


@app.on_event("shutdown")
async def shutdown_event():
    import logging

    logger = logging.getLogger("adip.health_api")
    logger.info("ADIP Health API shutting down")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("HEALTH_API_PORT", "8502"))
    uvicorn.run(
        "src.serving.health_api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True,
    )
