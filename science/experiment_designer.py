"""
science/experiment_designer.py  — v0.6
Experiment Auto-Designer.

Given a hypothesis verdict or business question, this module:
  1. Estimates the minimum detectable effect (MDE)
  2. Computes required sample size via power analysis
  3. Recommends experiment duration based on historical traffic
  4. Generates a full experiment spec the user can act on
  5. Stores the spec in OrgMemory for future hypothesis matching

This closes the full hypothesis → experiment → conclusion loop:
  HypothesisAgent generates hypotheses
  FeasibilityAgent marks them testable
  ConclusionEngine closes existing-data hypotheses
  ExperimentDesigner proposes new experiments for hypotheses that need more data

Design:
  - All stats via scipy (no external dependency)
  - LLM used only to write the human-readable spec (optional)
  - Deterministic power calc always runs even without LLM key
"""

from __future__ import annotations
import math
import uuid
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from core.logger import get_logger
from core.config import config

logger = get_logger(__name__)


@dataclass
class ExperimentSpec:
    spec_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis: str = ""
    metric: str = ""
    control_mean: float = 0.0
    expected_lift_pct: float = 5.0
    alpha: float = 0.05
    power: float = 0.80
    required_sample_per_variant: int = 0
    required_total_sample: int = 0
    estimated_duration_days: int = 0
    traffic_per_day: int = 0
    variants: list[str] = field(default_factory=lambda: ["control", "treatment"])
    recommended_guardrail_metrics: list[str] = field(default_factory=list)
    design_notes: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"Experiment: {self.hypothesis[:60]}\n"
            f"  Metric: {self.metric}  |  Expected lift: {self.expected_lift_pct:.1f}%\n"
            f"  Required sample/variant: {self.required_sample_per_variant:,}\n"
            f"  Estimated duration: {self.estimated_duration_days} days "
            f"(at {self.traffic_per_day:,}/day)\n"
            f"  α={self.alpha}  power={self.power}"
        )


class ExperimentDesigner:
    """
    Computes experiment specs from a hypothesis and current data context.
    Callable from ConclusionEngine (for inconclusive hypotheses) and
    directly from the UI.
    """

    def design(
        self,
        hypothesis: str,
        metric: str,
        df: pd.DataFrame,
        kpi_col: str,
        expected_lift_pct: float = 5.0,
        alpha: float = 0.05,
        power: float = 0.80,
        variants: list[str] = None,
        guardrail_metrics: list[str] = None,
    ) -> ExperimentSpec:
        """
        Main entry point. Returns a fully populated ExperimentSpec.
        """
        from datetime import datetime
        spec = ExperimentSpec(
            hypothesis=hypothesis,
            metric=metric,
            expected_lift_pct=expected_lift_pct,
            alpha=alpha,
            power=power,
            variants=variants or ["control", "treatment"],
            recommended_guardrail_metrics=guardrail_metrics or [],
            created_at=datetime.now().isoformat(),
        )

        # Step 1: Estimate baseline stats from current data
        baseline = self._estimate_baseline(df, kpi_col)
        spec.control_mean = baseline["mean"]

        # Step 2: Power analysis
        sample_size = self._compute_sample_size(
            mean=baseline["mean"],
            std=baseline["std"],
            lift_pct=expected_lift_pct,
            alpha=alpha,
            power=power,
        )
        spec.required_sample_per_variant = sample_size
        spec.required_total_sample = sample_size * len(spec.variants)

        # Step 3: Duration estimate from traffic
        traffic = self._estimate_daily_traffic(df, kpi_col)
        spec.traffic_per_day = traffic
        if traffic > 0:
            spec.estimated_duration_days = max(
                7,
                math.ceil(spec.required_total_sample / traffic)
            )
        else:
            spec.estimated_duration_days = -1   # unknown

        # Step 4: LLM-generated design notes (optional)
        spec.design_notes = self._llm_design_notes(spec) or self._rule_design_notes(spec)

        # Step 5: Save to org memory
        self._save_spec(spec)

        logger.info("ExperimentDesigner: %s", spec.summary())
        return spec

    def _estimate_baseline(self, df: pd.DataFrame, kpi_col: str) -> dict:
        if kpi_col not in df.columns:
            return {"mean": 0.0, "std": 1.0}
        series = pd.to_numeric(df[kpi_col], errors="coerce").dropna()
        if len(series) == 0:
            return {"mean": 0.0, "std": 1.0}
        return {
            "mean": float(series.mean()),
            "std":  float(series.std()) or 1.0,
        }

    def _compute_sample_size(
        self, mean: float, std: float, lift_pct: float,
        alpha: float, power: float
    ) -> int:
        """
        Two-sample t-test power analysis (Cohen's formula).
        Returns required n per variant.
        """
        try:
            from scipy.stats import norm
            delta   = abs(mean * lift_pct / 100) or (std * 0.05)
            effect  = delta / std if std > 0 else 0.05
            z_alpha = norm.ppf(1 - alpha / 2)
            z_beta  = norm.ppf(power)
            n = max(1, math.ceil(2 * ((z_alpha + z_beta) / effect) ** 2))
            return n
        except Exception as e:
            logger.warning("Power calc failed: %s — using 1000 default", e)
            return 1000

    def _estimate_daily_traffic(self, df: pd.DataFrame, kpi_col: str) -> int:
        """Estimate average daily volume from the dataset."""
        try:
            # Look for a date column
            date_cols = [c for c in df.columns if "date" in c.lower() or "day" in c.lower()]
            if date_cols:
                dates = pd.to_datetime(df[date_cols[0]], errors="coerce").dropna()
                if len(dates) >= 2:
                    days = max(1, (dates.max() - dates.min()).days)
                    return max(1, len(df) // days)
            # Fallback: median of the KPI col
            if kpi_col in df.columns:
                return max(1, int(pd.to_numeric(df[kpi_col], errors="coerce").median() or 100))
        except Exception:
            pass
        return 0

    def _llm_design_notes(self, spec: ExperimentSpec) -> str:
        if not (config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY):
            return ""
        try:
            from llm.client import LLMClient
            llm = LLMClient()
            prompt = f"""Given this experiment spec, write 3-4 concise design notes covering:
1. Key risks and confounders to watch
2. Recommended holdout/ramp strategy
3. When to declare significance vs stop early
4. Suggested guardrail metrics if not specified

Spec:
{spec.summary()}
Guardrails already specified: {spec.recommended_guardrail_metrics}

Write notes in plain bullets (max 150 words)."""
            return llm.complete(
                system="You are an experiment design expert. Be concise and specific.",
                user=prompt,
            )
        except Exception as e:
            logger.warning("LLM design notes failed: %s", e)
            return ""

    def _rule_design_notes(self, spec: ExperimentSpec) -> str:
        lines = []
        if spec.estimated_duration_days > 28:
            lines.append("⚠ Duration > 28 days: consider novelty effect, seasonality, and leakage.")
        if spec.expected_lift_pct < 2:
            lines.append("⚠ Lift < 2%: small effect requires large sample; monitor for p-hacking.")
        if not spec.recommended_guardrail_metrics:
            lines.append("💡 Add guardrail metrics (e.g. latency, error rate, revenue/user) before launch.")
        lines.append("📌 Use a pre-experiment AA test to validate randomisation before launch.")
        lines.append("📌 Freeze configuration and traffic allocation on launch day.")
        return "\n".join(lines)

    def _save_spec(self, spec: ExperimentSpec):
        try:
            from context_engine.org_memory import OrgMemory
            mem = OrgMemory()
            mem.save_insight(
                kpi=spec.metric,
                finding=f"Experiment spec created: {spec.hypothesis[:80]} "
                        f"[n={spec.required_sample_per_variant:,}, "
                        f"lift={spec.expected_lift_pct:.1f}%, "
                        f"duration={spec.estimated_duration_days}d]",
            )
        except Exception as e:
            logger.warning("Could not save experiment spec to org memory: %s", e)
