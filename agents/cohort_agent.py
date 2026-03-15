"""
agents/cohort_agent.py
Cohort Agent — auto-detects user + date columns,
builds retention matrix and time-to-convert if possible.
"""

from __future__ import annotations
import pandas as pd

from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from analysis.cohort_analyzer import CohortAnalyzer


class CohortAgent(BaseAgent):
    name = "cohort"
    description = "Retention matrix and time-to-convert from user activity data"

    def __init__(self):
        super().__init__()
        self._analyzer = CohortAnalyzer()

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        if df.empty:
            return self.skip("DataFrame is empty.")

        user_col = self._detect_user_col(df)
        date_col = self._detect_date_col(df, context.date_col)

        if not user_col:
            return self.skip("No user ID column detected.")
        if not date_col:
            return self.skip("No date column detected.")

        # Convert date column
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[df[date_col].notna() & df[user_col].notna()]

        if len(df) < 50:
            return self.skip(f"Only {len(df)} valid rows — need at least 50 for cohort analysis.")

        n_users = df[user_col].nunique()
        date_range_days = (df[date_col].max() - df[date_col].min()).days

        # Choose grain based on data range
        if date_range_days >= 90:
            grain = "M"
        elif date_range_days >= 14:
            grain = "W"
        else:
            grain = "D"

        # Build retention matrix
        matrix = None
        matrix_summary = ""
        try:
            matrix = self._analyzer.build_retention_matrix(
                df[[user_col, date_col]], user_col, date_col, grain
            )
            if not matrix.empty and "Period 1" in matrix.columns:
                # Average Period 1 retention across cohorts
                p1_avg = matrix["Period 1"].dropna().mean()
                matrix_summary = (
                    f"Avg Period-1 retention: {p1_avg:.1f}% "
                    f"across {len(matrix)} cohorts."
                )
        except Exception as e:
            self.logger.warning(f"Retention matrix failed: {e}")

        summary_parts = [
            f"Cohort analysis: {n_users:,} unique users, "
            f"{date_range_days} day range, grain={grain}."
        ]
        if matrix_summary:
            summary_parts.append(matrix_summary)

        return AgentResult(
            agent=self.name,
            status="success",
            summary=" ".join(summary_parts),
            data={
                "retention_matrix": matrix,
                "user_col": user_col,
                "date_col": date_col,
                "grain": grain,
                "n_users": n_users,
                "date_range_days": date_range_days,
            },
        )

    def _detect_user_col(self, df: pd.DataFrame) -> str | None:
        priority = ["user_id", "userid", "customer_id", "member_id",
                    "account_id", "user", "customer", "member"]
        for kw in priority:
            for col in df.columns:
                if kw == col.lower():
                    return col
        for col in df.columns:
            if any(kw in col.lower() for kw in ["user", "customer", "member", "account"]):
                return col
        return None

    def _detect_date_col(self, df: pd.DataFrame, context_date_col: str) -> str | None:
        if context_date_col and context_date_col in df.columns:
            return context_date_col
        for col in df.columns:
            if any(kw in col.lower() for kw in ["date", "time", "created", "joined", "signup"]):
                return col
        return None
