"""
insights/summary_builder.py
Assembles the full payload passed to InsightGenerator.
Combines analysis outputs into a structured dict.
"""

import pandas as pd
from core.logger import get_logger

logger = get_logger(__name__)


class SummaryBuilder:

    def build_exec_payload(
        self,
        kpi_col: str,
        delta: float,
        pct_change: float,
        anomalies: list[dict],
        top_drivers: list[dict],
        period_last: tuple = None,
        period_prev: tuple = None,
        comparisons: dict = None,
    ) -> dict:
        """
        Builds the structured payload for executive summary generation.
        Extended from app.py v0.1 payload dict.
        """
        payload = {
            "kpi": kpi_col,
            "overall_delta": round(delta, 2),
            "overall_pct_change": round(pct_change, 2),
            "anomalies_detected": len(anomalies),
            "anomaly_dates": [str(a.get("date", a.get("__date__", ""))) for a in anomalies[:5]],
            "top_negative_drivers": top_drivers[:3] if top_drivers else [],
            "top_positive_drivers": sorted(
                top_drivers, key=lambda x: x.get("delta", 0), reverse=True
            )[:3] if top_drivers else [],
        }
        if period_last:
            payload["period_last"] = f"{period_last[0]} to {period_last[1]}"
        if period_prev:
            payload["period_prev"] = f"{period_prev[0]} to {period_prev[1]}"
        if comparisons:
            payload["period_comparisons"] = comparisons

        logger.info("Exec payload built.")
        return payload

    def build_analysis_summary(
        self,
        kpi_col: str,
        ts: pd.DataFrame,
        value_col: str,
        driver_result: dict = None,
        anomaly_count: int = 0,
    ) -> str:
        """Short plain-text summary for use in follow-up question generation."""
        lines = [f"KPI: {kpi_col}"]
        if not ts.empty:
            lines.append(f"Data points: {len(ts)}")
            lines.append(f"Latest value: {ts[value_col].iloc[-1]:,.2f}")
            lines.append(f"Trend: {'up' if ts[value_col].iloc[-1] > ts[value_col].iloc[0] else 'down'}")
        if anomaly_count:
            lines.append(f"Anomalies detected: {anomaly_count}")
        if driver_result:
            lines.append(f"Overall change: {driver_result['delta']:+,.2f} ({driver_result['pct_change']:+.1f}%)")
        return " | ".join(lines)
