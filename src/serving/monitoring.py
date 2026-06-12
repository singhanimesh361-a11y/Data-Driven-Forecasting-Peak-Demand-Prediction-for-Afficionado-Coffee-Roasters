"""
Afficionado Demand Intelligence Platform — Monitoring Module
=============================================================
Production monitoring for forecast quality, model drift, data staleness,
and alerting via Slack / email.

Classes
-------
ForecastMonitor
    Tracks model drift, forecast availability, and data staleness.
AlertManager
    Sends alerts to Slack webhooks and/or email when thresholds are breached.
"""

from __future__ import annotations

import logging
import os
import smtplib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("adip.monitoring")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(
        logging.Formatter(
            '{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        )
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORES = ["Store_1", "Store_2", "Store_3"]
CATEGORIES = [
    "Coffee",
    "Tea",
    "Bakery",
    "Drinking Chocolate",
    "Coffee beans",
    "Branded",
    "Loose Tea",
    "Flavours",
    "Packaged Chocolate",
]

DRIFT_THRESHOLD_PP = 10.0  # percentage-point increase in MAPE triggers drift
STALENESS_HOURS = 25  # forecasts older than this are stale
MIN_FORECAST_ROWS = 100  # minimum rows expected in a forecast file


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------


@dataclass
class DriftResult:
    """Result of a drift check for one store×category pair."""

    store: str
    category: str
    training_mape: float
    recent_mape: float
    delta_pp: float
    is_drifted: bool
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store": self.store,
            "category": self.category,
            "training_mape": round(self.training_mape, 2),
            "recent_mape": round(self.recent_mape, 2),
            "delta_pp": round(self.delta_pp, 2),
            "is_drifted": self.is_drifted,
            "checked_at": self.checked_at,
        }


@dataclass
class HealthStatus:
    """Overall system health status."""

    healthy: bool
    forecasts_fresh: bool
    data_available: bool
    drift_detected: bool
    details: Dict[str, Any] = field(default_factory=dict)
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


# ---------------------------------------------------------------------------
# ForecastMonitor
# ---------------------------------------------------------------------------


class ForecastMonitor:
    """
    Monitors forecast quality, model drift, and data freshness.

    Parameters
    ----------
    forecast_dir : str | Path
        Directory containing forecast Parquet files.
    metrics_path : str | Path
        Path to the metrics/evaluation results (CSV or Parquet).
    training_metrics_path : str | Path
        Path to baseline training metrics for drift comparison.
    """

    def __init__(
        self,
        forecast_dir: str | Path = "data/forecast_store",
        metrics_path: str | Path = "data/processed/evaluation_metrics.csv",
        training_metrics_path: str | Path = "data/processed/training_metrics.csv",
    ) -> None:
        self.forecast_dir = Path(forecast_dir)
        self.metrics_path = Path(metrics_path)
        self.training_metrics_path = Path(training_metrics_path)

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def check_model_drift(self) -> List[DriftResult]:
        """
        Compare recent forecast accuracy (MAPE) against training baselines.

        A model is flagged as drifted if:
            recent_mape - training_mape > DRIFT_THRESHOLD_PP (10 pp)

        Returns
        -------
        list[DriftResult]
            Drift status for each store × category pair.
        """
        results: List[DriftResult] = []

        # Load training baselines
        training_metrics = self._load_training_metrics()

        # Load recent evaluation metrics
        recent_metrics = self._load_recent_metrics()

        for store in STORES:
            for category in CATEGORIES:
                key = (store, category)

                training_mape = training_metrics.get(key, 0.0)
                recent_mape = recent_metrics.get(key, training_mape)

                delta = recent_mape - training_mape
                is_drifted = delta > DRIFT_THRESHOLD_PP

                result = DriftResult(
                    store=store,
                    category=category,
                    training_mape=training_mape,
                    recent_mape=recent_mape,
                    delta_pp=delta,
                    is_drifted=is_drifted,
                )
                results.append(result)

                if is_drifted:
                    logger.warning(
                        "DRIFT DETECTED: %s / %s — MAPE increased %.1f pp " "(training=%.1f%%, recent=%.1f%%)",
                        store,
                        category,
                        delta,
                        training_mape,
                        recent_mape,
                    )

        drifted_count = sum(1 for r in results if r.is_drifted)
        logger.info(
            "Drift check complete: %d/%d pairs drifted",
            drifted_count,
            len(results),
        )
        return results

    def _load_training_metrics(self) -> Dict[tuple, float]:
        """Load baseline training MAPE values per (store, category)."""
        metrics: Dict[tuple, float] = {}

        if self.training_metrics_path.exists():
            df = pd.read_csv(self.training_metrics_path)
            for _, row in df.iterrows():
                key = (row.get("store_id", ""), row.get("product_category", ""))
                metrics[key] = float(row.get("mape", 0.0))
        else:
            # Default baselines when no file exists
            for store in STORES:
                for cat in CATEGORIES:
                    metrics[(store, cat)] = 15.0  # conservative default

            logger.warning(
                "Training metrics file not found at %s — using defaults",
                self.training_metrics_path,
            )

        return metrics

    def _load_recent_metrics(self) -> Dict[tuple, float]:
        """Load recent evaluation MAPE values."""
        metrics: Dict[tuple, float] = {}

        if self.metrics_path.exists():
            if self.metrics_path.suffix == ".csv":
                df = pd.read_csv(self.metrics_path)
            else:
                df = pd.read_parquet(self.metrics_path)
            for _, row in df.iterrows():
                key = (row.get("store_id", ""), row.get("product_category", ""))
                metrics[key] = float(row.get("mape", 0.0))
        else:
            logger.warning(
                "Recent metrics file not found at %s",
                self.metrics_path,
            )

        return metrics

    # ------------------------------------------------------------------
    # Forecast availability
    # ------------------------------------------------------------------

    def check_forecast_availability(self) -> Dict[str, Any]:
        """
        Check whether forecast files exist and contain sufficient data.

        Returns
        -------
        dict
            Availability status with file counts and row counts.
        """
        daily_latest = self.forecast_dir / "daily_forecast_latest.parquet"
        hourly_latest = self.forecast_dir / "hourly_forecast_latest.parquet"

        result: Dict[str, Any] = {
            "daily_exists": daily_latest.exists(),
            "hourly_exists": hourly_latest.exists(),
            "daily_rows": 0,
            "hourly_rows": 0,
            "daily_sufficient": False,
            "hourly_sufficient": False,
        }

        if daily_latest.exists():
            try:
                df = pd.read_parquet(daily_latest)
                result["daily_rows"] = len(df)
                result["daily_sufficient"] = len(df) >= MIN_FORECAST_ROWS
            except Exception as exc:
                logger.error("Failed to read daily forecast: %s", exc)

        if hourly_latest.exists():
            try:
                df = pd.read_parquet(hourly_latest)
                result["hourly_rows"] = len(df)
                result["hourly_sufficient"] = len(df) >= MIN_FORECAST_ROWS
            except Exception as exc:
                logger.error("Failed to read hourly forecast: %s", exc)

        available = result["daily_sufficient"] and result["hourly_sufficient"]
        result["available"] = available

        logger.info(
            "Forecast availability: %s (daily=%d rows, hourly=%d rows)",
            "OK" if available else "INSUFFICIENT",
            result["daily_rows"],
            result["hourly_rows"],
        )
        return result

    # ------------------------------------------------------------------
    # Data staleness
    # ------------------------------------------------------------------

    def check_data_staleness(self) -> Dict[str, Any]:
        """
        Check whether forecast data is stale (older than STALENESS_HOURS).

        Returns
        -------
        dict
            Staleness status with age in hours.
        """
        latest_path = self.forecast_dir / "daily_forecast_latest.parquet"

        if not latest_path.exists():
            return {
                "is_stale": True,
                "age_hours": None,
                "threshold_hours": STALENESS_HOURS,
                "message": "No forecast file found",
            }

        mtime = datetime.utcfromtimestamp(latest_path.stat().st_mtime)
        age_hours = (datetime.utcnow() - mtime).total_seconds() / 3600
        is_stale = age_hours > STALENESS_HOURS

        result = {
            "is_stale": is_stale,
            "age_hours": round(age_hours, 1),
            "threshold_hours": STALENESS_HOURS,
            "last_updated": mtime.isoformat() + "Z",
            "message": "STALE" if is_stale else "FRESH",
        }

        logger.info(
            "Data staleness: %s (age=%.1fh, threshold=%dh)",
            result["message"],
            age_hours,
            STALENESS_HOURS,
        )
        return result


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """
    Alert manager for sending notifications to Slack and email.

    Parameters
    ----------
    slack_webhook : str | None
        Slack incoming webhook URL.
    email : str | None
        Recipient email address for alert emails.
    smtp_host : str
        SMTP server hostname.
    smtp_port : int
        SMTP server port.
    smtp_user : str | None
        SMTP authentication username.
    smtp_password : str | None
        SMTP authentication password.
    """

    LEVEL_EMOJIS = {
        "info": "ℹ️",
        "warning": "⚠️",
        "critical": "🚨",
        "success": "✅",
    }

    LEVEL_COLORS = {
        "info": "#2196F3",
        "warning": "#FF9800",
        "critical": "#F44336",
        "success": "#4CAF50",
    }

    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        email: Optional[str] = None,
        smtp_host: str = "smtp.gmail.com",
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
    ) -> None:
        self.slack_webhook = slack_webhook or os.environ.get("ADIP_SLACK_WEBHOOK")
        self.email = email or os.environ.get("ADIP_ALERT_EMAIL")
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or os.environ.get("SMTP_USER")
        self.smtp_password = smtp_password or os.environ.get("SMTP_PASSWORD")

    def send_alert(self, level: str, title: str, body: str) -> None:
        """
        Send an alert via configured channels (Slack, email).

        Parameters
        ----------
        level : str
            One of: info, warning, critical, success
        title : str
            Alert title / subject line.
        body : str
            Alert body / detailed message.
        """
        level = level.lower()
        emoji = self.LEVEL_EMOJIS.get(level, "📋")
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        logger.info("Sending %s alert: %s", level.upper(), title)

        # --- Slack ---
        if self.slack_webhook:
            self._send_slack(level, emoji, title, body, timestamp)

        # --- Email ---
        if self.email and self.smtp_user:
            self._send_email(level, emoji, title, body, timestamp)

        if not self.slack_webhook and not self.email:
            logger.warning(
                "No alert channels configured — alert logged only: [%s] %s: %s",
                level.upper(),
                title,
                body,
            )

    def _send_slack(self, level: str, emoji: str, title: str, body: str, timestamp: str) -> None:
        """Send alert to Slack via incoming webhook."""
        color = self.LEVEL_COLORS.get(level, "#607D8B")
        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{emoji} ADIP Alert: {title}",
                            },
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": body,
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Level: *{level.upper()}* | {timestamp}",
                                }
                            ],
                        },
                    ],
                }
            ]
        }

        try:
            resp = requests.post(
                self.slack_webhook,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                logger.info("Slack alert sent successfully")
            else:
                logger.error(
                    "Slack alert failed: HTTP %d — %s",
                    resp.status_code,
                    resp.text,
                )
        except requests.RequestException as exc:
            logger.error("Slack alert failed: %s", exc)

    def _send_email(self, level: str, emoji: str, title: str, body: str, timestamp: str) -> None:
        """Send alert via SMTP email."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[ADIP {level.upper()}] {title}"
        msg["From"] = self.smtp_user
        msg["To"] = self.email

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: {self.LEVEL_COLORS.get(level, '#333')};">
                {emoji} ADIP Alert: {title}
            </h2>
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px;
                        border-left: 4px solid {self.LEVEL_COLORS.get(level, '#333')};">
                <pre style="white-space: pre-wrap;">{body}</pre>
            </div>
            <p style="color: #666; font-size: 12px; margin-top: 20px;">
                Level: {level.upper()} | {timestamp}<br>
                Afficionado Demand Intelligence Platform
            </p>
        </body>
        </html>
        """
        msg.attach(MIMEText(html_body, "html"))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)
            logger.info("Email alert sent to %s", self.email)
        except Exception as exc:
            logger.error("Email alert failed: %s", exc)

    def run_health_check(
        self,
        monitor: Optional[ForecastMonitor] = None,
    ) -> HealthStatus:
        """
        Run a comprehensive health check and alert if issues found.

        Parameters
        ----------
        monitor : ForecastMonitor, optional
            Monitor instance; creates a default one if not supplied.

        Returns
        -------
        HealthStatus
            Aggregated health status.
        """
        if monitor is None:
            monitor = ForecastMonitor()

        # 1. Check data staleness
        staleness = monitor.check_data_staleness()

        # 2. Check forecast availability
        availability = monitor.check_forecast_availability()

        # 3. Check model drift
        drift_results = monitor.check_model_drift()
        any_drift = any(r.is_drifted for r in drift_results)

        # Aggregate
        healthy = not staleness["is_stale"] and availability.get("available", False) and not any_drift

        status = HealthStatus(
            healthy=healthy,
            forecasts_fresh=not staleness["is_stale"],
            data_available=availability.get("available", False),
            drift_detected=any_drift,
            details={
                "staleness": staleness,
                "availability": availability,
                "drifted_models": [r.to_dict() for r in drift_results if r.is_drifted],
            },
        )

        # Send alerts for issues
        if staleness["is_stale"]:
            self.send_alert(
                "critical",
                "Forecast Data Stale",
                f"Forecasts are {staleness.get('age_hours', 'N/A')}h old "
                f"(threshold: {STALENESS_HOURS}h). Nightly refresh may have failed.",
            )

        if not availability.get("available", False):
            self.send_alert(
                "warning",
                "Forecast Availability Low",
                f"Daily rows: {availability['daily_rows']}, "
                f"Hourly rows: {availability['hourly_rows']}. "
                f"Minimum expected: {MIN_FORECAST_ROWS}.",
            )

        if any_drift:
            drifted = [r for r in drift_results if r.is_drifted]
            drift_summary = "\n".join(
                f"  • {r.store}/{r.category}: +{r.delta_pp:.1f}pp "
                f"(training={r.training_mape:.1f}%, recent={r.recent_mape:.1f}%)"
                for r in drifted
            )
            self.send_alert(
                "warning",
                f"Model Drift Detected ({len(drifted)} models)",
                f"The following models show MAPE increase > {DRIFT_THRESHOLD_PP}pp:\n{drift_summary}",
            )

        if healthy:
            logger.info("Health check PASSED — all systems nominal")
        else:
            logger.warning("Health check FAILED — see alerts above")

        return status
