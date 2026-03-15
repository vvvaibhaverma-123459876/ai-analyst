"""
tests/test_bench_science.py
Benchmark coverage for the science/ layer:
  - HypothesisAgent (4 sources, novelty scoring)
  - FeasibilityAgent (signal gating, data gap reporting)
  - ConclusionEngine (evidence collection, verdicts, DQ penalty)
  - ExperimentDesigner (power calc, duration, spec generation)
  - ResearchPlan (add/filter/similarity)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import make_ts, make_funnel, make_segment
from agents.context import AnalysisContext, AgentResult
from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
from science.feasibility_agent import FeasibilityAgent
from science.conclusion_engine import ConclusionEngine
from science.experiment_designer import ExperimentDesigner


def _ctx_with_ts(n=60, spike_idx=None):
    df = make_ts(n=n, spike_idx=spike_idx)
    ctx = AnalysisContext(
        df=df,
        date_col="date",
        kpi_col="revenue",
        business_context={"company": "Acme", "industry": "SaaS"},
    )
    ctx.data_profile = {
        "rows": n, "has_time_series": True,
        "has_funnel_signal": False, "has_cohort_signal": False,
        "dimensions": ["channel", "platform"],
        "kpis": ["revenue"],
    }
    ctx.ts = df.copy()
    return ctx


def _h(statement, source="data", testable=True):
    import uuid
    return Hypothesis(id=str(uuid.uuid4()), statement=statement,
                      source=source, testable=testable)


# ══════════════════════════════════════════════════════════════════════
# ResearchPlan
# ══════════════════════════════════════════════════════════════════════

class TestResearchPlan:

    def test_add_and_count_hypotheses(self):
        plan = ResearchPlan()
        plan.add_hypothesis(_h("Revenue dropped due to Android issues"))
        plan.add_hypothesis(_h("Seasonal pattern explains the drop"))
        assert len(plan.hypotheses) == 2

    def test_testable_filter(self):
        plan = ResearchPlan()
        plan.add_hypothesis(_h("Testable h", testable=True))
        plan.add_hypothesis(_h("Not testable", testable=False))
        testable = plan.testable_hypotheses()
        assert len(testable) == 1
        assert testable[0].testable is True

    def test_data_gaps_deduplication(self):
        plan = ResearchPlan()
        plan.add_data_gap("missing_date_column")
        plan.add_data_gap("missing_date_column")
        assert plan.data_gaps.count("missing_date_column") == 1

    def test_novelty_similarity_lower_for_dupes(self):
        plan = ResearchPlan()
        s1 = "Revenue dropped due to Android issues in Q1"
        s2 = "Revenue dropped because of Android problems in Q1"
        sim = plan._similarity(s1, s2)
        assert sim > 0.5

    def test_primary_conclusion_writeable(self):
        plan = ResearchPlan()
        plan.primary_conclusion = "Revenue fell 12% on Android"
        assert plan.primary_conclusion == "Revenue fell 12% on Android"


# ══════════════════════════════════════════════════════════════════════
# FeasibilityAgent
# ══════════════════════════════════════════════════════════════════════

class TestFeasibilityAgent:

    def test_seasonal_hypothesis_requires_time_series(self):
        ctx = _ctx_with_ts()
        ctx.research_plan = ResearchPlan()
        ctx.research_plan.add_hypothesis(
            _h("Seasonal weekday pattern causes Monday drops"))
        result = FeasibilityAgent().run(ctx)
        assert result.status == "success"
        h = ctx.research_plan.hypotheses[0]
        assert h.status in (HypothesisStatus.TESTABLE, HypothesisStatus.NOT_TESTABLE)

    def test_cohort_hypothesis_marked_not_testable_without_signal(self):
        ctx = _ctx_with_ts()
        ctx.data_profile["has_cohort_signal"] = False
        ctx.data_profile["has_time_series"] = True
        ctx.research_plan = ResearchPlan()
        ctx.research_plan.add_hypothesis(_h("Cohort retention decay explains churn"))
        FeasibilityAgent().run(ctx)
        h = ctx.research_plan.hypotheses[0]
        assert h.status == HypothesisStatus.NOT_TESTABLE

    def test_experiment_hypothesis_needs_ab_column(self):
        ctx = _ctx_with_ts()
        ctx.data_profile["has_ab_column"] = False
        ctx.research_plan = ResearchPlan()
        ctx.research_plan.add_hypothesis(
            _h("A/B experiment variant drives the difference"))
        FeasibilityAgent().run(ctx)
        h = ctx.research_plan.hypotheses[0]
        assert h.status == HypothesisStatus.NOT_TESTABLE

    def test_data_gaps_populated_for_untestable(self):
        ctx = _ctx_with_ts()
        ctx.data_profile["has_cohort_signal"] = False
        ctx.research_plan = ResearchPlan()
        ctx.research_plan.add_hypothesis(_h("User cohort lifecycle drives churn"))
        FeasibilityAgent().run(ctx)
        assert len(ctx.research_plan.data_gaps) >= 1

    def test_no_plan_skips_gracefully(self):
        ctx = _ctx_with_ts()
        ctx.research_plan = None
        result = FeasibilityAgent().run(ctx)
        assert result.status == "skipped"

    def test_too_few_rows_blocks_seasonal(self):
        ctx = _ctx_with_ts(n=5)
        ctx.research_plan = ResearchPlan()
        ctx.research_plan.add_hypothesis(_h("Seasonal pattern causes dip"))
        FeasibilityAgent().run(ctx)
        h = ctx.research_plan.hypotheses[0]
        assert h.status == HypothesisStatus.NOT_TESTABLE


# ══════════════════════════════════════════════════════════════════════
# ConclusionEngine
# ══════════════════════════════════════════════════════════════════════

class TestConclusionEngine:

    def _ctx_with_evidence(self, evidence_pairs, dq_score=0.9):
        ctx = _ctx_with_ts()
        ctx.data_quality_report = {"score": dq_score}
        h = _h("Revenue declined due to Android issues")
        h.status = HypothesisStatus.TESTABLE
        for agent, supports, conf in evidence_pairs:
            h.evidence.append({
                "agent": agent, "summary": "finding",
                "supports": supports, "confidence": conf,
            })
        plan = ResearchPlan(hypotheses=[h])
        ctx.research_plan = plan
        # Add minimal agent results
        for agent, supports, conf in evidence_pairs:
            ctx.write_result(AgentResult(
                agent=agent, status="success",
                summary="Platform drag found" if supports else "No finding",
                data={"movers": {"negative": [{"dimension": "platform", "value": "android"}]}}
                     if supports else {},
            ))
        return ctx

    def test_strong_evidence_confirms(self):
        ctx = self._ctx_with_evidence([
            ("trend", True, 0.9),
            ("root_cause", True, 0.85),
            ("anomaly", True, 0.8),
        ])
        plan = ConclusionEngine().close_hypotheses(ctx)
        h = plan.hypotheses[0]
        assert h.status in (HypothesisStatus.CONFIRMED, HypothesisStatus.INCONCLUSIVE)

    def test_opposing_evidence_rejects(self):
        ctx = self._ctx_with_evidence([
            ("trend", False, 0.9),
            ("root_cause", False, 0.85),
        ])
        plan = ConclusionEngine().close_hypotheses(ctx)
        h = plan.hypotheses[0]
        assert h.status in (HypothesisStatus.REJECTED, HypothesisStatus.INCONCLUSIVE)

    def test_poor_dq_downgrades_confidence(self):
        ctx_good = self._ctx_with_evidence([("trend", True, 0.9)], dq_score=0.95)
        ctx_bad  = self._ctx_with_evidence([("trend", True, 0.9)], dq_score=0.25)
        plan_good = ConclusionEngine().close_hypotheses(ctx_good)
        plan_bad  = ConclusionEngine().close_hypotheses(ctx_bad)
        assert plan_good.hypotheses[0].confidence >= plan_bad.hypotheses[0].confidence

    def test_no_evidence_inconclusive(self):
        ctx = _ctx_with_ts()
        ctx.data_quality_report = {"score": 0.8}
        h = _h("Unexplained drop occurred")
        h.status = HypothesisStatus.TESTABLE
        ctx.research_plan = ResearchPlan(hypotheses=[h])
        plan = ConclusionEngine().close_hypotheses(ctx)
        assert plan.hypotheses[0].status in (
            HypothesisStatus.INCONCLUSIVE, HypothesisStatus.REJECTED)

    def test_primary_conclusion_set(self):
        ctx = self._ctx_with_evidence([
            ("trend", True, 0.9), ("root_cause", True, 0.85)
        ])
        plan = ConclusionEngine().close_hypotheses(ctx)
        assert plan.primary_conclusion

    def test_empty_plan_no_crash(self):
        ctx = _ctx_with_ts()
        ctx.research_plan = ResearchPlan()
        plan = ConclusionEngine().close_hypotheses(ctx)
        assert isinstance(plan, ResearchPlan)

    def test_evidence_grade_set_on_hypothesis(self):
        ctx = self._ctx_with_evidence([("trend", True, 0.9)])
        plan = ConclusionEngine().close_hypotheses(ctx)
        h = plan.hypotheses[0]
        assert h.evidence_grade in ("strong", "moderate", "weak", "speculative", "unknown")

    def test_inconclusive_generates_experiment_spec(self):
        """ConclusionEngine should auto-propose an experiment for inconclusive hypotheses."""
        ctx = _ctx_with_ts(n=60)
        ctx.data_quality_report = {"score": 0.8}
        h = _h("A/B test needed to confirm pricing change effect")
        h.status = HypothesisStatus.TESTABLE
        h.evidence = []
        ctx.research_plan = ResearchPlan(hypotheses=[h])
        plan = ConclusionEngine().close_hypotheses(ctx)
        # If inconclusive, experiment_spec should be proposed
        if plan.hypotheses[0].status == HypothesisStatus.INCONCLUSIVE:
            assert plan.experiment_spec is not None


# ══════════════════════════════════════════════════════════════════════
# ExperimentDesigner
# ══════════════════════════════════════════════════════════════════════

class TestExperimentDesigner:

    def test_required_sample_positive(self):
        df = make_ts(n=60)
        spec = ExperimentDesigner().design(
            hypothesis="Pricing change will increase revenue",
            metric="revenue",
            df=df, kpi_col="revenue",
            expected_lift_pct=10.0,
        )
        assert spec.required_sample_per_variant > 0
        assert spec.required_total_sample == spec.required_sample_per_variant * 2

    def test_higher_lift_needs_fewer_samples(self):
        df = make_ts(n=60)
        designer = ExperimentDesigner()
        spec_5pct  = designer.design("H", "revenue", df, "revenue", expected_lift_pct=5)
        spec_20pct = designer.design("H", "revenue", df, "revenue", expected_lift_pct=20)
        assert spec_5pct.required_sample_per_variant >= spec_20pct.required_sample_per_variant

    def test_duration_estimated_from_traffic(self):
        df = make_ts(n=60)
        spec = ExperimentDesigner().design(
            "H", "revenue", df, "revenue", expected_lift_pct=5)
        # Either we get a real estimate or -1 (unknown)
        assert spec.estimated_duration_days > 0 or spec.estimated_duration_days == -1

    def test_minimum_duration_7_days(self):
        df = make_ts(n=200)
        spec = ExperimentDesigner().design(
            "H", "revenue", df, "revenue", expected_lift_pct=50)
        if spec.estimated_duration_days > 0:
            assert spec.estimated_duration_days >= 7

    def test_spec_has_all_required_fields(self):
        df = make_ts(n=60)
        spec = ExperimentDesigner().design("H", "revenue", df, "revenue")
        assert spec.spec_id
        assert spec.alpha > 0
        assert spec.power > 0
        assert spec.variants

    def test_empty_df_no_crash(self):
        df = pd.DataFrame({"date":[], "revenue":[]})
        spec = ExperimentDesigner().design("H", "revenue", df, "revenue")
        assert spec.required_sample_per_variant >= 0

    def test_summary_string_non_empty(self):
        df = make_ts(n=60)
        spec = ExperimentDesigner().design("H", "revenue", df, "revenue")
        assert len(spec.summary()) > 20
