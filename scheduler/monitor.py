"""
scheduler/monitor.py  — v0.6
Proactive analytics monitor.

Runs on a schedule (APScheduler), pulls data from the active connector,
runs the anomaly detection pipeline, and dispatches alerts — with zero
human initiation.

Architecture:
  SchedulerConfig (configs/schedule.yaml) →
  MonitorJob (per-KPI, per-connector) →
  AgentRunner (anomaly + conclusion only, lightweight mode) →
  AlertDispatcher (Slack / email / in-app)

The scheduler is completely separate from the Streamlit UI.
Run standalone:
    python -m scheduler.monitor

Or start from the FastAPI service (if ENABLE_SCHEDULER=true).
"""

from __future__ import annotations
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
import pandas as pd

from core.logger import get_logger
from core.config import config

logger = get_logger(__name__)

SCHEDULE_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "schedule.yaml"
ENABLE_SCHEDULER = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"


# ------------------------------------------------------------------
# Job configuration dataclass
# ------------------------------------------------------------------

@dataclass
class MonitorJob:
    job_id: str
    name: str
    connector: str          # "postgres" | "snowflake" | "bigquery" | "redshift" | "athena"
    query: str              # SQL to pull the latest KPI data
    kpi_col: str
    date_col: str
    cron: str = "0 * * * *"    # every hour by default
    alert_channels: list[str] = field(default_factory=lambda: ["slack", "in_app"])
    tenant_id: str = "default"
    enabled: bool = True
    last_run: str = ""
    last_status: str = "pending"


# ------------------------------------------------------------------
# Schedule loader
# ------------------------------------------------------------------

def load_schedule() -> list[MonitorJob]:
    """Load monitor jobs from configs/schedule.yaml."""
    if not SCHEDULE_CONFIG_PATH.exists():
        return _default_jobs()
    try:
        import yaml
        with open(SCHEDULE_CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
        jobs = []
        for j in raw.get("jobs", []):
            jobs.append(MonitorJob(
                job_id        = j.get("id", str(uuid.uuid4())),
                name          = j.get("name", "Unnamed job"),
                connector     = j.get("connector", "postgres"),
                query         = j.get("query", ""),
                kpi_col       = j.get("kpi_col", ""),
                date_col      = j.get("date_col", ""),
                cron          = j.get("cron", "0 * * * *"),
                alert_channels= j.get("alert_channels", ["slack", "in_app"]),
                tenant_id     = j.get("tenant_id", "default"),
                enabled       = j.get("enabled", True),
            ))
        logger.info("Loaded %d monitor jobs from schedule.yaml", len(jobs))
        return jobs
    except Exception as e:
        logger.warning("Could not load schedule.yaml: %s", e)
        return []


def _default_jobs() -> list[MonitorJob]:
    return []


# ------------------------------------------------------------------
# Core monitor execution
# ------------------------------------------------------------------

class MonitorRunner:
    """Executes a single MonitorJob: pulls data → analyses → alerts."""

    def __init__(self, on_alert: Callable[[str, str, str], None] = None):
        self._on_alert = on_alert   # callback(job_name, message, urgency)

    def run_job(self, job: MonitorJob) -> dict:
        logger.info("Running monitor job: %s", job.name)
        t0 = time.time()
        result = {
            "job_id":   job.job_id,
            "job_name": job.name,
            "run_id":   str(uuid.uuid4()),
            "started":  datetime.now().isoformat(),
            "status":   "error",
            "anomaly_count": 0,
            "alert_sent": False,
            "message":  "",
        }

        try:
            df = self._fetch_data(job)
            if df.empty:
                result["status"] = "skipped"
                result["message"] = "No data returned by query"
                return result

            anomaly_result = self._detect_anomalies(df, job)
            result["anomaly_count"] = anomaly_result.get("anomaly_count", 0)

            if result["anomaly_count"] > 0:
                message = self._build_alert_message(job, df, anomaly_result)
                self._dispatch_alert(job, message)
                result["alert_sent"] = True
                result["message"] = message[:200]

            result["status"] = "ok"
            result["elapsed_sec"] = round(time.time() - t0, 2)

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            logger.error("Monitor job %s failed: %s", job.name, e)

        job.last_run = datetime.now().isoformat()
        job.last_status = result["status"]
        return result

    def _fetch_data(self, job: MonitorJob) -> pd.DataFrame:
        from connectors.registry import ConnectorRegistry
        registry = ConnectorRegistry()
        if job.connector in registry.available():
            registry.set_active(job.connector)
        return registry.execute(job.query)

    def _detect_anomalies(self, df: pd.DataFrame, job: MonitorJob) -> dict:
        """Lightweight anomaly detection (no full pipeline — fast for scheduling)."""
        try:
            from analysis.anomaly_detector import AnomalyDetector
            detector = AnomalyDetector()
            ts = df.copy()
            if job.date_col and job.date_col in ts.columns:
                ts[job.date_col] = pd.to_datetime(ts[job.date_col], errors="coerce")
                ts = ts.sort_values(job.date_col)
            result = detector.detect(
                ts,
                kpi_col=job.kpi_col,
                date_col=job.date_col or None,
            )
            return result
        except Exception as e:
            logger.warning("Anomaly detection in scheduler failed: %s", e)
            return {"anomaly_count": 0, "anomaly_records": []}

    def _build_alert_message(self, job: MonitorJob, df: pd.DataFrame,
                             anomaly_result: dict) -> str:
        records = anomaly_result.get("anomaly_records", [])
        n = anomaly_result.get("anomaly_count", 0)
        lines = [
            f"🔔 Anomaly detected — {job.name}",
            f"KPI: {job.kpi_col}  |  {n} anomalous point(s)",
        ]
        for r in records[:3]:
            date = str(r.get("date", ""))[:10]
            val  = r.get("value", "?")
            z    = r.get("z_score", "?")
            lines.append(f"  • {date}: value={val}  z={z}")
        lines.append(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M')} UTC")
        return "\n".join(lines)

    def _dispatch_alert(self, job: MonitorJob, message: str):
        try:
            from output.alert_dispatcher import AlertDispatcher
            dispatcher = AlertDispatcher()
            dispatcher.dispatch(channels=job.alert_channels, message=message, urgency="high")
            if self._on_alert:
                self._on_alert(job.name, message, "high")
        except Exception as e:
            logger.warning("Alert dispatch failed: %s", e)


# ------------------------------------------------------------------
# APScheduler wrapper
# ------------------------------------------------------------------

class AnalyticsScheduler:
    """
    Wraps APScheduler for proactive monitoring.

    Usage:
        scheduler = AnalyticsScheduler()
        scheduler.start()
        # ... runs in background
        scheduler.stop()
    """

    def __init__(self, on_alert: Callable[[str, str, str], None] = None):
        self._runner = MonitorRunner(on_alert=on_alert)
        self._scheduler = None
        self._jobs: list[MonitorJob] = []

    def load_jobs(self) -> list[MonitorJob]:
        self._jobs = load_schedule()
        return self._jobs

    def start(self):
        if not ENABLE_SCHEDULER:
            logger.info("Scheduler disabled (ENABLE_SCHEDULER=false)")
            return
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            from apscheduler.triggers.cron import CronTrigger
            self._scheduler = BackgroundScheduler(timezone="UTC")
            self.load_jobs()
            for job in self._jobs:
                if not job.enabled:
                    continue
                try:
                    parts = job.cron.split()
                    trigger = CronTrigger(
                        minute=parts[0], hour=parts[1],
                        day=parts[2] if len(parts) > 2 else "*",
                        month=parts[3] if len(parts) > 3 else "*",
                        day_of_week=parts[4] if len(parts) > 4 else "*",
                    )
                    self._scheduler.add_job(
                        func=self._runner.run_job,
                        trigger=trigger,
                        args=[job],
                        id=job.job_id,
                        name=job.name,
                        replace_existing=True,
                    )
                    logger.info("Scheduled job '%s' [%s]", job.name, job.cron)
                except Exception as e:
                    logger.warning("Failed to schedule job %s: %s", job.name, e)

            self._scheduler.start()
            logger.info("AnalyticsScheduler started with %d jobs", len(self._jobs))

        except ImportError:
            logger.warning(
                "APScheduler not installed — scheduler disabled. "
                "Run: pip install apscheduler"
            )
        except Exception as e:
            logger.error("Scheduler start failed: %s", e)

    def stop(self):
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("AnalyticsScheduler stopped")

    def run_job_now(self, job_id: str) -> dict:
        """Manually trigger a job by ID (for testing / on-demand)."""
        for job in self._jobs:
            if job.job_id == job_id:
                return self._runner.run_job(job)
        return {"error": f"Job {job_id} not found"}

    def status(self) -> dict:
        running = self._scheduler is not None and self._scheduler.running
        return {
            "running": running,
            "jobs": [
                {
                    "id": j.job_id, "name": j.name,
                    "cron": j.cron, "enabled": j.enabled,
                    "last_run": j.last_run, "last_status": j.last_status,
                }
                for j in self._jobs
            ],
        }


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import signal
    scheduler = AnalyticsScheduler()
    scheduler.start()
    logger.info("Scheduler running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.stop()
        logger.info("Scheduler stopped.")
