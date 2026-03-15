"""
agents/anomaly_agent.py
Anomaly Agent — auto-selects best detection method based on data length,
runs detection, and returns flagged points with severity.
"""

from __future__ import annotations
import pandas as pd

from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from analysis.anomaly_detector import AnomalyDetector
from core.config import config


class AnomalyAgent(BaseAgent):
    name = "anomaly"
    description = "Detects outliers using Z-score, IQR, or STL — auto-selected by data length"

    def _run(self, context: AnalysisContext) -> AgentResult:
        ts = context.ts
        date_col = context.date_col
        kpi_col = context.kpi_col

        if ts.empty:
            return self.skip("No time series in context — run Trend Agent first.")
        if not kpi_col or kpi_col not in ts.columns:
            return self.skip(f"KPI column '{kpi_col}' not in time series.")

        n = len(ts)
        detector = AnomalyDetector(
            window=config.DEFAULT_ROLLING_WINDOW,
            z_threshold=config.DEFAULT_Z_THRESHOLD,
        )

        # Auto-select method based on data length
        if n >= 60:
            method = "STL"
            ts_anom = detector.detect_stl(ts, date_col, kpi_col)
        elif n >= 20:
            method = "Z-Score"
            ts_anom = detector.detect_zscore(ts, date_col, kpi_col)
        else:
            method = "IQR"
            ts_anom = detector.detect_iqr(ts, kpi_col)

        anomaly_records = detector.summarise(ts_anom, date_col, kpi_col)
        anom_count = len(anomaly_records)

        # Severity breakdown
        severity_counts = {"high": 0, "medium": 0, "low": 0}
        if "severity" in ts_anom.columns:
            for sev, cnt in ts_anom[ts_anom["anomaly"] == True]["severity"].value_counts().items():
                if sev in severity_counts:
                    severity_counts[sev] = int(cnt)

        # Write enriched ts back to context
        context.ts = ts_anom

        if anom_count == 0:
            summary = f"No anomalies detected using {method} on {n} data points."
        else:
            dates = [str(a["date"])[:10] for a in anomaly_records[:3]]
            summary = (
                f"{anom_count} anomal{'y' if anom_count == 1 else 'ies'} detected "
                f"({method}, {n} points). "
                f"High severity: {severity_counts['high']}, "
                f"Medium: {severity_counts['medium']}. "
                f"Dates: {', '.join(dates)}{'...' if len(anomaly_records) > 3 else ''}."
            )

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "ts_with_anomalies": ts_anom,
                "anomaly_records": anomaly_records,
                "anomaly_count": anom_count,
                "severity_counts": severity_counts,
                "method_used": method,
                "data_points": n,
            },
        )
