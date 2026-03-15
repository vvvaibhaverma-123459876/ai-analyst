"""
agents/root_cause_agent.py
Root Cause Agent — driver attribution + contribution analysis across all dimensions.
Uses the most recent period vs prior period automatically.
"""

from __future__ import annotations
import pandas as pd

from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from analysis.root_cause import RootCauseAnalyzer
from core.config import config


class RootCauseAgent(BaseAgent):
    name = "root_cause"
    description = "Drivers, contribution analysis, segment-level delta breakdown"

    def __init__(self):
        super().__init__()
        self._analyzer = RootCauseAnalyzer()

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        date_col = context.date_col
        kpi_col = context.kpi_col

        if not date_col or not kpi_col:
            return self.skip("No date or KPI column in context.")
        if df.empty:
            return self.skip("DataFrame is empty.")

        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df_valid = df[df[date_col].notna()]

        if df_valid.empty:
            return self.skip("No valid rows after date parsing.")

        # Driver attribution
        result = self._analyzer.driver_attribution(df_valid, date_col, kpi_col,
                                                    config.DEFAULT_DRIVER_DAYS)
        delta = result["delta"]
        pct = result["pct_change"]
        drivers = result["drivers"]
        movers = self._analyzer.top_movers(drivers, n=5)

        # Contribution analysis — run for each dimension
        dimensions = context.data_profile.get("dimensions", [])
        contributions = {}
        for dim in dimensions[:4]:   # cap at 4 dims to keep it fast
            try:
                contributions[dim] = self._analyzer.contribution_analysis(
                    df_valid, kpi_col, dim, date_col
                )
            except Exception as e:
                self.logger.warning(f"Contribution for '{dim}' failed: {e}")

        # Build summary sentence
        direction = "increased" if delta > 0 else "decreased"
        top_neg = movers["negative"][:1]
        top_pos = movers["positive"][:1]

        summary_parts = [
            f"KPI {direction} by {delta:+,.0f} ({pct:+.1f}%) "
            f"comparing last {config.DEFAULT_DRIVER_DAYS} days vs prior period."
        ]
        if top_neg:
            d = top_neg[0]
            summary_parts.append(
                f"Biggest drag: {d['dimension']}={d['value']} "
                f"({d['delta']:+,.0f}, {d.get('pct_contribution', 0):.1f}% of total change)."
            )
        if top_pos:
            d = top_pos[0]
            summary_parts.append(
                f"Biggest lift: {d['dimension']}={d['value']} ({d['delta']:+,.0f})."
            )

        return AgentResult(
            agent=self.name,
            status="success",
            summary=" ".join(summary_parts),
            data={
                "delta": delta,
                "pct_change": pct,
                "last_total": result["last_total"],
                "prev_total": result["prev_total"],
                "drivers": drivers,
                "movers": movers,
                "contributions": contributions,
                "period_last": result["period_last"],
                "period_prev": result["period_prev"],
            },
        )
