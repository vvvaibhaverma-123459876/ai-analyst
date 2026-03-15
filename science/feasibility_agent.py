"""
science/feasibility_agent.py
Feasibility Agent — filters hypotheses to only those testable with current data.

For each hypothesis:
  1. Checks required columns exist
  2. Checks minimum row counts
  3. Assigns analysis agents responsible for testing it
  4. Documents what data is missing for untestable ones

Output: ResearchPlan with hypotheses marked TESTABLE or NOT_TESTABLE,
        and a data_gaps list the user can act on.
"""

from __future__ import annotations
import uuid
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
from core.logger import get_logger

logger = get_logger(__name__)

# Keywords that map hypothesis topics to required data signals
HYPOTHESIS_REQUIREMENTS = {
    "pipeline|etl|ingestion|data error|duplicate": {
        "agents": ["anomaly"],
        "required_signals": [],
        "min_rows": 10,
    },
    "seasonal|weekday|weekend|cyclical|pattern": {
        "agents": ["trend", "anomaly"],
        "required_signals": ["has_time_series"],
        "min_rows": 28,
    },
    "segment|channel|platform|source|device": {
        "agents": ["root_cause"],
        "required_signals": ["has_dimensions"],
        "min_rows": 20,
    },
    "experiment|a/b|test|variant|treatment|control": {
        "agents": ["experiment"],
        "required_signals": ["has_ab_column"],
        "min_rows": 30,
    },
    "funnel|conversion|drop.off|step|stage": {
        "agents": ["funnel"],
        "required_signals": ["has_funnel_signal"],
        "min_rows": 20,
    },
    "cohort|retention|churn|user lifecycle": {
        "agents": ["cohort"],
        "required_signals": ["has_cohort_signal"],
        "min_rows": 50,
    },
    "cluster|segment|group|type of user": {
        "agents": ["ml_cluster"],
        "required_signals": ["has_numeric_features"],
        "min_rows": 20,
    },
    "forecast|predict|future|next week|next month": {
        "agents": ["forecast"],
        "required_signals": ["has_time_series"],
        "min_rows": 14,
    },
}


class FeasibilityAgent(BaseAgent):
    name = "feasibility"
    description = "Filters hypotheses to testable ones and assigns analysis agents"

    def _run(self, context: AnalysisContext) -> AgentResult:
        plan: ResearchPlan = getattr(context, "research_plan", None)
        if plan is None or not plan.hypotheses:
            return self.skip("No research plan or hypotheses to assess.")

        df = context.df
        profile = context.data_profile
        rows = len(df)

        # Build signal map from current data
        signals = self._build_signal_map(df, profile)

        testable_count = 0
        untestable_count = 0

        for h in plan.hypotheses:
            result = self._assess_hypothesis(h, signals, rows, context)
            h.status = result["status"]
            h.testable = result["testable"]
            h.assigned_agents = result["agents"]
            h.missing_data = result["missing_data"]
            if result["missing_data"]:
                for gap in result["missing_data"]:
                    plan.add_data_gap(gap)
            if result["testable"]:
                h.status = HypothesisStatus.TESTABLE
                testable_count += 1
            else:
                h.status = HypothesisStatus.NOT_TESTABLE
                untestable_count += 1

        summary = (
            f"Feasibility: {testable_count} testable, {untestable_count} not testable. "
            f"Data gaps: {len(plan.data_gaps)}. "
        )
        if plan.data_gaps:
            summary += f"Missing: {', '.join(plan.data_gaps[:3])}"

        return AgentResult(
            agent=self.name, status="success",
            summary=summary,
            data={
                "testable_count": testable_count,
                "untestable_count": untestable_count,
                "data_gaps": plan.data_gaps,
                "hypothesis_statuses": [
                    {"id": h.id, "statement": h.statement[:60],
                     "testable": h.testable, "agents": h.assigned_agents}
                    for h in plan.hypotheses
                ],
            },
        )

    def _build_signal_map(self, df, profile: dict) -> dict:
        import numpy as np
        signals = {}
        signals["has_time_series"] = profile.get("has_time_series", False)
        signals["has_funnel_signal"] = profile.get("has_funnel_signal", False)
        signals["has_cohort_signal"] = profile.get("has_cohort_signal", False)
        signals["has_dimensions"] = bool(profile.get("dimensions"))
        signals["has_numeric_features"] = (
            len(df.select_dtypes(include=[np.number]).columns) >= 2
        )
        ab_kws = ["variant", "group", "treatment", "control", "experiment", "arm"]
        signals["has_ab_column"] = any(
            any(kw in c.lower() for kw in ab_kws)
            for c in df.columns
        )
        return signals

    def _assess_hypothesis(
        self, h: Hypothesis, signals: dict, rows: int, context: AnalysisContext
    ) -> dict:
        import re
        stmt_lower = h.statement.lower()
        missing = list(h.missing_data)  # start with any already noted

        for pattern, req in HYPOTHESIS_REQUIREMENTS.items():
            if re.search(pattern, stmt_lower):
                # Check row count
                if rows < req["min_rows"]:
                    missing.append(f"Need ≥{req['min_rows']} rows (have {rows})")
                    return {"testable": False, "status": HypothesisStatus.NOT_TESTABLE,
                            "agents": [], "missing_data": missing}
                # Check signals
                for sig in req["required_signals"]:
                    if not signals.get(sig):
                        missing.append(f"Missing data signal: {sig.replace('has_', '').replace('_', ' ')}")
                        return {"testable": False, "status": HypothesisStatus.NOT_TESTABLE,
                                "agents": [], "missing_data": missing}
                # All checks passed
                return {"testable": True, "status": HypothesisStatus.TESTABLE,
                        "agents": req["agents"], "missing_data": missing}

        # No pattern matched — general testability
        if rows >= 10 and (signals.get("has_time_series") or signals.get("has_dimensions")):
            return {"testable": True, "status": HypothesisStatus.TESTABLE,
                    "agents": ["root_cause", "trend"], "missing_data": missing}

        missing.append("Insufficient data or unrecognised hypothesis type")
        return {"testable": False, "status": HypothesisStatus.NOT_TESTABLE,
                "agents": [], "missing_data": missing}
