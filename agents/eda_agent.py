"""
agents/eda_agent.py
EDA Agent — runs first, always.
Profiles the DataFrame, infers KPIs + dimensions, detects date column,
and populates context.data_profile for the Orchestrator to use.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from analysis.eda_engine import EDAEngine
from connectors.csv_connector import CSVConnector


class EDAAgent(BaseAgent):
    name = "eda"
    description = "Profiles data, detects column types, suggests KPIs and dimensions"

    def __init__(self):
        super().__init__()
        self._engine = EDAEngine()

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        if df.empty:
            return self.skip("DataFrame is empty")

        profile_df = self._engine.profile(df)
        quality = self._engine.quality_report(df)
        kpis = self._engine.infer_kpis(df)
        dimensions = self._engine.infer_dimensions(df)
        date_col = CSVConnector.detect_datetime_column(df)

        has_time_series = date_col is not None
        has_funnel_signal = self._detect_funnel_signal(df)
        has_cohort_signal = self._detect_cohort_signal(df)

        # Write to context so Orchestrator can read without re-running
        context.data_profile = {
            "rows": quality["total_rows"],
            "cols": quality["total_columns"],
            "kpis": kpis,
            "dimensions": dimensions,
            "date_col": date_col,
            "has_time_series": has_time_series,
            "has_funnel_signal": has_funnel_signal,
            "has_cohort_signal": has_cohort_signal,
            "completeness_pct": quality["completeness_pct"],
            "duplicate_rows": quality["duplicate_rows"],
        }

        # Set context fields if not already set
        if not context.date_col and date_col:
            context.date_col = date_col
        if not context.kpi_col and kpis:
            context.kpi_col = kpis[0]

        summary = (
            f"Dataset: {quality['total_rows']:,} rows, {quality['total_columns']} columns. "
            f"Suggested KPIs: {', '.join(kpis[:3]) or 'none'}. "
            f"Dimensions: {', '.join(dimensions[:4]) or 'none'}. "
            f"Time series: {'yes' if has_time_series else 'no'}. "
            f"Completeness: {quality['completeness_pct']}%."
        )

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "profile_df": profile_df,
                "quality": quality,
                "kpis": kpis,
                "dimensions": dimensions,
                "date_col": date_col,
                "has_time_series": has_time_series,
                "has_funnel_signal": has_funnel_signal,
                "has_cohort_signal": has_cohort_signal,
            },
        )

    def _detect_funnel_signal(self, df: pd.DataFrame) -> bool:
        """True if a column looks like it contains funnel stage names."""
        funnel_keywords = ["stage", "step", "event", "funnel", "status", "phase"]
        for col in df.select_dtypes(include="object").columns:
            if any(kw in col.lower() for kw in funnel_keywords):
                return True
            # Check values for stage-like patterns
            if df[col].nunique() <= 15:
                vals = " ".join(df[col].dropna().unique().astype(str)).lower()
                if any(kw in vals for kw in ["signup", "kyc", "payment", "convert",
                                              "activate", "onboard", "verify"]):
                    return True
        return False

    def _detect_cohort_signal(self, df: pd.DataFrame) -> bool:
        """True if dataset has user IDs and dates — enough for cohort analysis."""
        has_user_col = any(
            any(kw in c.lower() for kw in ["user", "customer", "member", "account"])
            for c in df.columns
        )
        has_date = any(
            "date" in c.lower() or "time" in c.lower()
            for c in df.columns
        )
        return has_user_col and has_date
