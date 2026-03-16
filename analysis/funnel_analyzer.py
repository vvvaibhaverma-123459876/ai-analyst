"""
analysis/funnel_analyzer.py  — v10
Funnel analysis: step conversion, drop-off rates, stage-wise comparison.

v10: implements AnalysisContract — analyze() is the canonical entry point.
     All existing methods preserved for backward-compat with agents.
"""

import pandas as pd
import numpy as np
from analysis.contract import AnalysisContract, AnalysisResult
from core.logger import get_logger

logger = get_logger(__name__)


class FunnelAnalyzer(AnalysisContract):

    module_name = "funnel"

    # ------------------------------------------------------------------
    # AnalysisContract: canonical entry point
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, **kwargs) -> AnalysisResult:
        """
        Contract entry point.  kwargs accepted:
          stage_col, user_col, stages (list), date_col, date_filter
        Returns AnalysisResult with funnel DataFrame in metadata["funnel_df"].
        """
        stage_col  = kwargs.get("stage_col", "")
        user_col   = kwargs.get("user_col", "")
        stages     = kwargs.get("stages")
        date_col   = kwargs.get("date_col")
        date_filter= kwargs.get("date_filter")

        required = [c for c in [stage_col, user_col] if c]
        errors = self.validate_inputs(df, required_cols=required, min_rows=2)
        if errors:
            return AnalysisResult(
                module=self.module_name, ok=False, errors=errors,
                summary="Funnel analysis could not run.", method="funnel",
            )

        try:
            funnel_df = self.compute_funnel(
                df, stage_col, user_col, stages, date_col, date_filter
            )
            drop = self.biggest_drop(funnel_df)
            n_stages    = len(funnel_df)
            top_users   = int(funnel_df.iloc[0]["users"]) if not funnel_df.empty else 0
            bottom_rate = float(funnel_df.iloc[-1]["conversion_from_top_pct"]) if not funnel_df.empty else 0.0
            summary = (
                f"Funnel: {n_stages} stages, {top_users} top-of-funnel users, "
                f"{bottom_rate:.1f}% end-to-end conversion."
            )
            if drop:
                summary += f" Biggest drop at '{drop['stage']}' ({drop['drop_off_pct']:.1f}% lost)."

            return AnalysisResult(
                module=self.module_name,
                ok=True,
                records=funnel_df.to_dict("records"),
                summary=summary,
                confidence=1.0,
                method="funnel",
                metadata={
                    "funnel_df":  funnel_df,
                    "biggest_drop": drop,
                    "n_stages":   n_stages,
                    "top_users":  top_users,
                    "end_to_end_pct": bottom_rate,
                },
            )
        except Exception as e:
            return AnalysisResult(
                module=self.module_name, ok=False,
                errors=[str(e)], summary=f"Funnel error: {e}", method="funnel",
            )

    def compute_funnel(
        self,
        df: pd.DataFrame,
        stage_col: str,
        user_col: str,
        stages: list[str] = None,
        date_col: str = None,
        date_filter: tuple = None,
    ) -> pd.DataFrame:
        """
        Computes funnel conversion for ordered stages.

        Args:
            df: DataFrame with event-level data
            stage_col: column containing stage names
            user_col: column containing unique user IDs
            stages: ordered list of stage names to include
            date_col: optional date column for filtering
            date_filter: optional (start_date, end_date) tuple

        Returns:
            DataFrame with columns:
            [stage, users, conversion_from_prev, conversion_from_top, drop_off_pct]
        """
        if date_col and date_filter:
            start, end = date_filter
            df = df[(df[date_col] >= start) & (df[date_col] <= end)]

        if stages is None:
            stages = df[stage_col].dropna().unique().tolist()

        rows = []
        prev_users = None
        top_users = None

        for stage in stages:
            users = df[df[stage_col] == stage][user_col].nunique()
            if top_users is None:
                top_users = users

            conv_from_prev = (users / prev_users * 100) if prev_users else 100.0
            conv_from_top = (users / top_users * 100) if top_users else 100.0
            drop_off = 100.0 - conv_from_prev

            rows.append({
                "stage": stage,
                "users": users,
                "conversion_from_prev_pct": round(conv_from_prev, 1),
                "conversion_from_top_pct": round(conv_from_top, 1),
                "drop_off_pct": round(max(drop_off, 0), 1),
            })
            prev_users = users

        result = pd.DataFrame(rows)
        logger.info(f"Funnel computed: {len(stages)} stages, top={top_users} users")
        return result

    def compute_aggregated_funnel(
        self,
        df: pd.DataFrame,
        stages: list[str],
        counts: list[int],
    ) -> pd.DataFrame:
        """
        Compute funnel from pre-aggregated stage → user counts.
        Useful when df is already summarised.
        """
        rows = []
        top = counts[0] if counts else 1
        prev = None
        for stage, count in zip(stages, counts):
            conv_prev = (count / prev * 100) if prev else 100.0
            conv_top = (count / top * 100) if top else 100.0
            rows.append({
                "stage": stage,
                "users": count,
                "conversion_from_prev_pct": round(conv_prev, 1),
                "conversion_from_top_pct": round(conv_top, 1),
                "drop_off_pct": round(max(100 - conv_prev, 0), 1),
            })
            prev = count
        return pd.DataFrame(rows)

    def biggest_drop(self, funnel_df: pd.DataFrame) -> dict:
        """Returns the stage with the largest drop-off."""
        if funnel_df.empty or len(funnel_df) < 2:
            return {}
        idx = funnel_df.iloc[1:]["drop_off_pct"].idxmax()
        row = funnel_df.loc[idx]
        return {
            "stage": row["stage"],
            "drop_off_pct": row["drop_off_pct"],
            "users_lost": funnel_df.loc[idx - 1, "users"] - row["users"]
            if idx > 0 else 0,
        }

    def compare_funnels(
        self,
        df: pd.DataFrame,
        stage_col: str,
        user_col: str,
        segment_col: str,
        stages: list[str],
    ) -> pd.DataFrame:
        """
        Computes funnel separately for each value in segment_col.
        Returns a wide DataFrame: stage × segment conversion rates.
        """
        segments = df[segment_col].dropna().unique()
        all_results = []
        for seg in segments:
            seg_df = df[df[segment_col] == seg]
            funnel = self.compute_funnel(seg_df, stage_col, user_col, stages)
            funnel["segment"] = seg
            all_results.append(funnel)

        combined = pd.concat(all_results)
        pivot = combined.pivot_table(
            index="stage",
            columns="segment",
            values="conversion_from_top_pct",
        )
        return pivot.reindex(stages)
