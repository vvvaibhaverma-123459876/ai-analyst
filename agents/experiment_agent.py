"""
agents/experiment_agent.py
Experiment Agent — statistical testing for causal attribution.

Detects:
  - A/B test columns (variant, group, treatment, control, experiment)
  - Pre/post event patterns (did a product change cause the metric shift?)
  - Natural experiments (diff-in-diff when two groups exist)

Runs:
  - Two-sample t-test (continuous KPI)
  - Chi-squared test (conversion / binary KPI)
  - Diff-in-diff (when treatment + time can be identified)
  - MDE (minimum detectable effect) calculator
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult

AB_KEYWORDS = ["variant", "group", "treatment", "control", "experiment",
               "cohort", "arm", "bucket", "test", "holdout", "version"]


class ExperimentAgent(BaseAgent):
    name = "experiment"
    description = "A/B test analysis, t-tests, chi-squared, diff-in-diff"

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        kpi_col = context.kpi_col

        if df.empty or not kpi_col:
            return self.skip("No data or KPI column.")

        ab_col = self._detect_ab_column(df)
        if ab_col is None:
            return self.skip("No A/B test or experiment column detected.")

        groups = df[ab_col].dropna().unique()
        if len(groups) < 2:
            return self.skip(f"Column '{ab_col}' has fewer than 2 groups.")

        # Use first two groups for primary test
        g_a = groups[0]
        g_b = groups[1]
        vals_a = df[df[ab_col] == g_a][kpi_col].dropna()
        vals_b = df[df[ab_col] == g_b][kpi_col].dropna()

        if len(vals_a) < 5 or len(vals_b) < 5:
            return self.skip("Fewer than 5 observations per group — test unreliable.")

        results = {}

        # ── T-test ───────────────────────────────────────────────────
        t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        cohens_d = (vals_b.mean() - vals_a.mean()) / (
            np.sqrt((vals_a.std() ** 2 + vals_b.std() ** 2) / 2) + 1e-9
        )
        lift = (vals_b.mean() - vals_a.mean()) / (vals_a.mean() + 1e-9) * 100

        results["ttest"] = {
            "group_a": str(g_a), "group_b": str(g_b),
            "mean_a": round(float(vals_a.mean()), 4),
            "mean_b": round(float(vals_b.mean()), 4),
            "lift_pct": round(lift, 2),
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_val, 4),
            "significant": bool(p_val < 0.05),
            "cohens_d": round(cohens_d, 3),
            "n_a": len(vals_a), "n_b": len(vals_b),
        }

        # ── Chi-squared (if binary/conversion KPI) ───────────────────
        is_binary = set(df[kpi_col].dropna().unique()).issubset({0, 1, True, False, "0", "1"})
        chi2_result = None
        if is_binary:
            ct = pd.crosstab(df[ab_col], df[kpi_col])
            if ct.shape == (2, 2):
                chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
                chi2_result = {
                    "chi2": round(chi2, 4),
                    "p_value": round(p_chi, 4),
                    "significant": bool(p_chi < 0.05),
                    "degrees_of_freedom": dof,
                }
                results["chi2"] = chi2_result

        # ── Diff-in-diff (if date col available) ─────────────────────
        did_result = None
        if context.date_col and context.date_col in df.columns:
            did_result = self._diff_in_diff(df, ab_col, kpi_col, context.date_col, g_a, g_b)
            if did_result:
                results["diff_in_diff"] = did_result

        # ── MDE calculation ──────────────────────────────────────────
        pooled_std = np.sqrt((vals_a.var() + vals_b.var()) / 2)
        mde = 1.96 * pooled_std * np.sqrt(2 / min(len(vals_a), len(vals_b))) * 100
        results["mde_pct"] = round(mde, 2)

        # ── Summary ─────────────────────────────────────────────────
        sig = results["ttest"]["significant"]
        direction = "higher" if lift > 0 else "lower"
        conf = "statistically significant" if sig else "NOT statistically significant"
        summary = (
            f"A/B test on '{ab_col}': {g_b} vs {g_a}. "
            f"{kpi_col} is {abs(lift):.1f}% {direction} in {g_b} — {conf} "
            f"(p={p_val:.3f}, n={len(vals_a)}+{len(vals_b)}). "
            f"Cohen's d={cohens_d:.2f} ({'large' if abs(cohens_d)>0.8 else 'medium' if abs(cohens_d)>0.5 else 'small'} effect)."
        )
        if did_result:
            summary += f" Diff-in-diff: treatment effect = {did_result.get('treatment_effect', 0):+.2f}."

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "ab_col": ab_col,
                "group_a": str(g_a),
                "group_b": str(g_b),
                "results": results,
                "significant": sig,
                "lift_pct": round(lift, 2),
                "p_value": round(p_val, 4),
                "cohens_d": round(cohens_d, 3),
            },
        )

    def _detect_ab_column(self, df: pd.DataFrame) -> str | None:
        for col in df.select_dtypes(include="object").columns:
            if any(kw in col.lower() for kw in AB_KEYWORDS):
                if 2 <= df[col].nunique() <= 10:
                    return col
        for col in df.select_dtypes(include="object").columns:
            if df[col].nunique() == 2:
                vals = set(df[col].dropna().str.lower().unique())
                if vals & {"control", "treatment", "a", "b", "test", "holdout"}:
                    return col
        return None

    def _diff_in_diff(
        self, df, ab_col, kpi_col, date_col, g_a, g_b
    ) -> dict | None:
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])
            mid = df[date_col].median()
            df["post"] = (df[date_col] > mid).astype(int)
            df["treated"] = (df[ab_col] == g_b).astype(int)

            pre_control  = df[(df["post"] == 0) & (df["treated"] == 0)][kpi_col].mean()
            post_control = df[(df["post"] == 1) & (df["treated"] == 0)][kpi_col].mean()
            pre_treat    = df[(df["post"] == 0) & (df["treated"] == 1)][kpi_col].mean()
            post_treat   = df[(df["post"] == 1) & (df["treated"] == 1)][kpi_col].mean()

            did = (post_treat - pre_treat) - (post_control - pre_control)
            return {
                "pre_control": round(pre_control, 2),
                "post_control": round(post_control, 2),
                "pre_treatment": round(pre_treat, 2),
                "post_treatment": round(post_treat, 2),
                "treatment_effect": round(did, 2),
                "midpoint_date": str(mid.date()),
            }
        except Exception as e:
            self.logger.warning(f"DiD failed: {e}")
            return None
