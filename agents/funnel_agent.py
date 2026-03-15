"""
agents/funnel_agent.py
Funnel Agent — auto-detects stage column + user column,
computes conversion funnel, identifies biggest drop-off.
"""

from __future__ import annotations
import pandas as pd

from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from analysis.funnel_analyzer import FunnelAnalyzer
from core.config import config


# Stage name keywords used for auto-detection
STAGE_KEYWORDS = [
    "signup", "register", "kyc", "verify", "payment", "pay",
    "convert", "activate", "onboard", "complete", "submit",
    "start", "initiat", "success", "fail", "pending",
]


class FunnelAgent(BaseAgent):
    name = "funnel"
    description = "Conversion funnel: step-by-step drop-off and segment comparison"

    def __init__(self):
        super().__init__()
        self._analyzer = FunnelAnalyzer()

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        if df.empty:
            return self.skip("DataFrame is empty.")

        stage_col = self._detect_stage_col(df)
        user_col = self._detect_user_col(df)

        if not stage_col:
            return self.skip("No funnel stage column detected.")
        if not user_col:
            return self.skip("No user ID column detected.")

        stages = self._order_stages(df[stage_col].dropna().unique().tolist())
        if len(stages) < 2:
            return self.skip("Fewer than 2 distinct stages — cannot build funnel.")

        funnel_df = self._analyzer.compute_funnel(df, stage_col, user_col, stages)
        biggest_drop = self._analyzer.biggest_drop(funnel_df)

        top_users = funnel_df["users"].iloc[0] if not funnel_df.empty else 0
        bottom_users = funnel_df["users"].iloc[-1] if not funnel_df.empty else 0
        overall_conv = round(bottom_users / top_users * 100, 1) if top_users else 0

        summary_parts = [
            f"Funnel: {len(stages)} stages, overall conversion {overall_conv}% "
            f"({top_users:,} → {bottom_users:,} users)."
        ]
        if biggest_drop:
            summary_parts.append(
                f"Biggest drop at '{biggest_drop['stage']}': "
                f"{biggest_drop['drop_off_pct']:.1f}% drop-off "
                f"({biggest_drop['users_lost']:,} users lost)."
            )

        return AgentResult(
            agent=self.name,
            status="success",
            summary=" ".join(summary_parts),
            data={
                "funnel_df": funnel_df,
                "stage_col": stage_col,
                "user_col": user_col,
                "stages": stages,
                "overall_conversion_pct": overall_conv,
                "biggest_drop": biggest_drop,
                "top_of_funnel_users": int(top_users),
                "bottom_of_funnel_users": int(bottom_users),
            },
        )

    # ------------------------------------------------------------------
    # Auto-detection helpers
    # ------------------------------------------------------------------

    def _detect_stage_col(self, df: pd.DataFrame) -> str | None:
        """Find the column most likely to contain stage/event names."""
        for col in df.select_dtypes(include="object").columns:
            if any(kw in col.lower() for kw in ["stage", "step", "event", "funnel", "status"]):
                return col
        # Fallback: look at values
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() <= 20:
                vals = " ".join(df[col].dropna().unique().astype(str)).lower()
                if sum(1 for kw in STAGE_KEYWORDS if kw in vals) >= 2:
                    return col
        return None

    def _detect_user_col(self, df: pd.DataFrame) -> str | None:
        """Find a user identifier column."""
        for col in df.columns:
            if any(kw in col.lower() for kw in ["user_id", "userid", "user id",
                                                  "customer_id", "member_id", "account_id"]):
                return col
        for col in df.columns:
            if any(kw in col.lower() for kw in ["user", "customer", "member", "account"]):
                return col
        return None

    def _order_stages(self, stages: list[str]) -> list[str]:
        """
        Try to sort stages in logical funnel order using keyword priority.
        Falls back to alphabetical if no keywords match.
        """
        priority = [
            "signup", "register", "onboard", "kyc", "verify",
            "initiat", "start", "submit", "payment", "pay",
            "convert", "complete", "success", "activat",
        ]
        def stage_rank(s: str) -> int:
            sl = s.lower()
            for i, kw in enumerate(priority):
                if kw in sl:
                    return i
            return len(priority)

        return sorted(stages, key=stage_rank)
