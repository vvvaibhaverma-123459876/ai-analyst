"""
agents/trend_agent.py
Trend Agent — resamples the KPI time series, computes DoD/WoW/MoM,
adds trend line and rolling bands.
"""

from __future__ import annotations
import pandas as pd

from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from analysis.statistics import (
    resample_timeseries, period_comparison, add_trend_line, rolling_stats
)
from core.config import config
from semantic.grain_resolver import GrainResolver
from semantic.metric_registry import MetricRegistry


class TrendAgent(BaseAgent):
    name = "trend"
    description = "Time series analysis: resampling, trend line, MoM/WoW/DoD comparisons"

    def __init__(self):
        super().__init__()
        self._metric_registry = MetricRegistry()
        self._grain_resolver = GrainResolver(self._metric_registry)

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        date_col = context.date_col
        kpi_col = context.kpi_col

        if not date_col or not kpi_col:
            return self.skip("No date or KPI column set in context.")
        if df.empty:
            return self.skip("DataFrame is empty.")

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col].notna()]

        if df.empty:
            return self.skip("No valid rows after date parsing.")

        metric_key = self._resolve_metric_key(context, kpi_col)
        resolved_grain = self._resolve_runtime_grain(metric_key, context.grain)
        context.grain = resolved_grain

        # Build time series
        ts = resample_timeseries(df, date_col, kpi_col, resolved_grain)
        ts = rolling_stats(ts, kpi_col, window=config.DEFAULT_ROLLING_WINDOW)
        ts = add_trend_line(ts, date_col, kpi_col)

        # Write back to context for other agents
        context.ts = ts

        # Period comparisons
        comparisons = {}
        for comp in ["DoD", "WoW", "MoM"]:
            try:
                comparisons[comp] = period_comparison(df, date_col, kpi_col, comp)
            except Exception:
                pass

        # Trend direction
        if len(ts) >= 2:
            first_val = ts[kpi_col].iloc[0]
            last_val = ts[kpi_col].iloc[-1]
            trend_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
            direction = "upward" if trend_pct > 2 else ("downward" if trend_pct < -2 else "flat")
        else:
            trend_pct = 0
            direction = "insufficient data"

        # Seasonality check (basic: std of weekly means)
        seasonality_note = ""
        try:
            if len(ts) >= 14:
                ts_temp = ts.copy()
                ts_temp["dow"] = pd.to_datetime(ts_temp[date_col]).dt.dayofweek
                dow_var = ts_temp.groupby("dow")[kpi_col].mean().std()
                overall_std = ts_temp[kpi_col].std()
                if overall_std > 0 and (dow_var / overall_std) > 0.2:
                    seasonality_note = "Day-of-week pattern detected."
        except Exception:
            pass

        summary_parts = [
            f"KPI '{kpi_col}' shows a {direction} trend ({trend_pct:+.1f}% overall) at {resolved_grain} grain."
        ]
        for comp_name, comp in comparisons.items():
            summary_parts.append(f"{comp_name}: {comp['pct_change']:+.1f}%")
        if seasonality_note:
            summary_parts.append(seasonality_note)

        summary = " | ".join(summary_parts)

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "ts": ts,
                "comparisons": comparisons,
                "trend_direction": direction,
                "trend_pct": round(trend_pct, 2),
                "seasonality_note": seasonality_note,
                "resolved_grain": resolved_grain,
                "latest_value": round(ts[kpi_col].iloc[-1], 2) if not ts.empty else None,
            },
        )

    def _resolve_metric_key(self, context: AnalysisContext, kpi_col: str) -> str | None:
        candidates = [
            context.business_context.get("metric") if isinstance(context.business_context, dict) else None,
            context.business_context.get("kpi") if isinstance(context.business_context, dict) else None,
            kpi_col,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return self._metric_registry.get(str(candidate)).key
            except Exception:
                resolved = self._metric_registry.resolve(str(candidate))
                if resolved:
                    return resolved
        return None

    def _resolve_runtime_grain(self, metric_key: str | None, requested_grain: str | None) -> str:
        if not metric_key:
            return self._normalise_for_timeseries(requested_grain)
        semantic_grain = self._grain_resolver.resolve(metric_key, requested_grain)
        return self._normalise_for_timeseries(semantic_grain)

    @staticmethod
    def _normalise_for_timeseries(grain: str | None) -> str:
        if not grain:
            return "Daily"
        mapping = {
            "hourly": "Daily",
            "daily": "Daily",
            "weekly": "Weekly",
            "monthly": "Monthly",
            "day": "Daily",
            "week": "Weekly",
            "month": "Monthly",
            "Daily": "Daily",
            "Weekly": "Weekly",
            "Monthly": "Monthly",
        }
        return mapping.get(str(grain), str(grain))
