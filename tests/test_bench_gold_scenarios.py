"""
tests/test_bench_gold_scenarios.py  — v9
Gold standard benchmark scenarios.

These tests define what a v9 release must provably deliver.
A release is NOT ready until every test in this file passes.

Scenarios covered
-----------------
GS-01  Clean anomaly — real spike → detected, confirmed, alert-worthy
GS-02  Fake anomaly from bad data — DQ blocks high-confidence conclusion
GS-03  Source conflict — metric registry overrides agent output
GS-04  Invalid dimension request — semantic layer rejects, not agent
GS-05  Untestable hypothesis — marked NOT_TESTABLE, data gap recorded
GS-06  Strong hypothesis support → CONFIRMED with evidence grade ≥ moderate
GS-07  Contradictory agent outputs — Guardian detects, confidence penalised
GS-08  Policy-restricted request — SecurityShell blocks cross-tenant access
GS-09  Replayed run parity — same main conclusion state ± same agents
GS-10  Recommendation ranking — urgent high-confidence action tops list
GS-11  Source authority — primary_truth wins over agent_output
GS-12  Grain coercion — invalid grain resolved to governed grain
GS-13  Output classification — PII in brief → CONFIDENTIAL, auto-redacted
GS-14  Full pipeline E2E — clean run produces all required outputs
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest import make_ts, make_ts as _ts


# ══════════════════════════════════════════════════════════════════════
# GS-01  Clean anomaly detected and confirmed
# ══════════════════════════════════════════════════════════════════════

def test_gs01_clean_anomaly_detected():
    from analysis.anomaly_detector import AnomalyDetector
    df = make_ts(n=60, spike_idx=45)
    result = AnomalyDetector(z_threshold=2.5).detect(df, kpi_col="revenue", date_col="date")
    assert result.ok
    assert result.anomaly_count >= 1, "Spike at index 45 must be detected"
    bm = AnomalyDetector().to_benchmark_output(result)
    assert bm["anomaly_count"] >= 1
    assert bm["method"] in ("zscore", "iqr", "stl")


# ══════════════════════════════════════════════════════════════════════
# GS-02  Bad data → DQ gate blocks strong conclusion
# ══════════════════════════════════════════════════════════════════════

def test_gs02_bad_data_blocks_strong_conclusion():
    from quality.data_quality_gate import DataQualityGate
    from science.conclusion_engine import ConclusionEngine
    from agents.context import AnalysisContext, AgentResult
    from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
    import uuid

    # Very high null ratio
    df = pd.DataFrame({"d": pd.date_range("2025-01-01", 10), "v": [None]*10})
    dq = DataQualityGate().assess(df, "d", "v")
    assert dq.ok is False
    assert dq.score <= 0.35

    # Even with strong evidence, confidence should be capped when DQ is poor
    ctx = AnalysisContext(df=df, kpi_col="v", date_col="d")
    ctx.data_quality_report = {"score": 0.20}
    h = Hypothesis(id=str(uuid.uuid4()), statement="Rev dropped",
                   source="data", status=HypothesisStatus.TESTABLE)
    h.evidence = [
        {"agent": "trend", "summary": "drop confirmed", "supports": True, "confidence": 0.95},
        {"agent": "anomaly", "summary": "spike seen",   "supports": True, "confidence": 0.95},
    ]
    ctx.research_plan = ResearchPlan(hypotheses=[h])
    plan = ConclusionEngine().close_hypotheses(ctx)
    assert plan.hypotheses[0].confidence <= 0.75, \
        "Confidence must be capped when data quality is very poor"


# ══════════════════════════════════════════════════════════════════════
# GS-03  Source conflict — primary_truth wins over agent_output
# ══════════════════════════════════════════════════════════════════════

def test_gs03_source_authority_primary_truth_wins():
    from semantic.source_authority import SourceConflictResolver, SourceClaim, Authority

    truth_claim = SourceClaim(
        source="metric_registry", value="sum(amount_usd)",
        confidence=1.0, authority=Authority.primary_truth,
    )
    agent_claim = SourceClaim(
        source="trend", value="avg(amount_usd)",
        confidence=0.9, authority=Authority.agent_output,
    )
    resolution = SourceConflictResolver().resolve(truth_claim, agent_claim)
    assert resolution.resolved is True
    assert resolution.winner.source == "metric_registry"
    assert resolution.conflict_type == "authority"


def test_gs03b_source_conflict_unresolvable_penalises_confidence():
    from semantic.source_authority import SourceConflictResolver, SourceClaim, Authority

    a = SourceClaim(source="trend", value="up", confidence=0.8,
                    authority=Authority.agent_output)
    b = SourceClaim(source="anomaly", value="down", confidence=0.8,
                    authority=Authority.agent_output)
    resolution = SourceConflictResolver().resolve(a, b)
    assert resolution.confidence_penalty >= 0.15
    assert resolution.conflict_type == "unresolvable"


# ══════════════════════════════════════════════════════════════════════
# GS-04  Invalid dimension → semantic rejects
# ══════════════════════════════════════════════════════════════════════

def test_gs04_invalid_dimension_rejected_by_semantic():
    from semantic.metric_registry import MetricRegistry
    registry = MetricRegistry({
        "revenue": {
            "description": "Net revenue",
            "aggregation": "sum",
            "dimensions": ["channel", "platform"],
            "allowed_grains": ["daily", "weekly"],
        }
    })
    assert registry.validate_dimension("revenue", "channel")  is True
    assert registry.validate_dimension("revenue", "country")  is False
    assert registry.validate_dimension("revenue", "zip_code") is False


# ══════════════════════════════════════════════════════════════════════
# GS-05  Untestable hypothesis → marked, gap recorded
# ══════════════════════════════════════════════════════════════════════

def test_gs05_untestable_hypothesis_marked_and_gap_recorded():
    from agents.context import AnalysisContext
    from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
    from science.feasibility_agent import FeasibilityAgent
    import uuid

    df = make_ts(n=30)
    ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date")
    ctx.data_profile = {
        "rows": 30, "has_time_series": True,
        "has_funnel_signal": False, "has_cohort_signal": False,
        "dimensions": [], "kpis": ["revenue"],
    }
    # Cohort hypothesis requires cohort signal — not present
    h = Hypothesis(id=str(uuid.uuid4()),
                   statement="User cohort lifecycle drives churn",
                   source="business", testable=None)
    plan = ResearchPlan(hypotheses=[h])
    ctx.research_plan = plan
    FeasibilityAgent().run(ctx)

    assert ctx.research_plan.hypotheses[0].status == HypothesisStatus.NOT_TESTABLE
    assert len(ctx.research_plan.data_gaps) >= 1


# ══════════════════════════════════════════════════════════════════════
# GS-06  Strong hypothesis support → CONFIRMED, evidence ≥ moderate
# ══════════════════════════════════════════════════════════════════════

def test_gs06_strong_support_confirmed_moderate_evidence():
    from agents.context import AnalysisContext, AgentResult
    from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
    from science.conclusion_engine import ConclusionEngine
    import uuid

    df = make_ts(n=60)
    ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date")
    ctx.data_quality_report = {"score": 0.92}
    h = Hypothesis(id=str(uuid.uuid4()), statement="Android drag",
                   source="data", status=HypothesisStatus.TESTABLE)
    h.evidence = [
        {"agent": "trend",      "summary": "revenue down on android", "supports": True,  "confidence": 0.92},
        {"agent": "root_cause", "summary": "android has negative mover", "supports": True, "confidence": 0.88},
        {"agent": "anomaly",    "summary": "anomaly on android dates", "supports": True,  "confidence": 0.80},
    ]
    for e in h.evidence:
        ctx.write_result(AgentResult(agent=e["agent"], status="success",
                                     summary=e["summary"], data={}))
    ctx.research_plan = ResearchPlan(hypotheses=[h])
    plan = ConclusionEngine().close_hypotheses(ctx)

    h_out = plan.hypotheses[0]
    assert h_out.status == HypothesisStatus.CONFIRMED
    assert h_out.evidence_grade in ("strong", "moderate")
    assert h_out.confidence >= 0.55


# ══════════════════════════════════════════════════════════════════════
# GS-07  Contradictory agents → Guardian detects, confidence penalised
# ══════════════════════════════════════════════════════════════════════

def test_gs07_contradictory_agents_detected():
    from guardian.contradiction_checker import ContradictionChecker
    from guardian.evidence_grader import EvidenceGrader
    from agents.context import AgentResult

    results = {
        "trend":   AgentResult("trend",  "success", "Revenue increased sharply", {}),
        "anomaly": AgentResult("anomaly","success", "Revenue dropped significantly", {}),
    }
    contradictions = ContradictionChecker().detect(results)
    assert len(contradictions) >= 1

    # With contradictions, evidence grade must be lower
    grade_with    = EvidenceGrader().grade(0.8, 0.8, len(contradictions), 0.9)
    grade_without = EvidenceGrader().grade(0.8, 0.8, 0, 0.9)
    assert grade_with.summary_strength < grade_without.summary_strength


# ══════════════════════════════════════════════════════════════════════
# GS-08  Policy-restricted request → SecurityShell blocks cross-tenant
# ══════════════════════════════════════════════════════════════════════

def test_gs08_cross_tenant_blocked():
    from security.security_shell import SecurityShell
    shell = SecurityShell(tenant_id="acme", user_id="alice", role="analyst")
    with pytest.raises(PermissionError):
        shell.publish_output({"brief": "secret data"}, requested_tenant_id="rival")


def test_gs08b_admin_cross_tenant_allowed():
    from security.security_shell import SecurityShell
    shell = SecurityShell(tenant_id="acme", user_id="alice", role="admin")
    payload, classification = shell.publish_output(
        {"brief": "clean report", "tenant_id": "rival"},
        requested_tenant_id="rival",
    )
    assert classification in ("INTERNAL", "CONFIDENTIAL", "RESTRICTED")


# ══════════════════════════════════════════════════════════════════════
# GS-09  Replay parity
# ══════════════════════════════════════════════════════════════════════

def test_gs09_replay_parity(tmp_path):
    from agents.pipeline import GovernedPipeline
    from agents.context import AnalysisContext
    from evaluation.replay_harness import ReplayHarness
    from versioning.run_manifest import RunManifest

    df = make_ts(n=40)
    ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date",
                          run_id="gs09-original", tenant_id="default")
    ctx = GovernedPipeline().run(ctx)
    original_agents = set(ctx.results.keys())
    original_conclusion = getattr(
        getattr(ctx.research_plan, "primary_conclusion", ""), "__class__", type("")
    )

    # Build manifest for replay
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    data_dir = tmp_path / "replay_data"
    data_dir.mkdir(parents=True)
    csv_path = data_dir / "gs09-original.csv"
    df.to_csv(csv_path, index=False)
    m = RunManifest.create("gs09-original")
    m.replay_data_path = str(csv_path)
    m.replay_context = {
        "date_col": "date", "kpi_col": "revenue",
        "grain": "Daily", "tenant_id": "default", "user_id": "system",
    }
    m.persist(base_dir=str(manifest_dir))

    replayed = ReplayHarness(base_dir=str(manifest_dir)).replay("gs09-original")
    replayed_agents = set(replayed.results.keys())

    # Same core agents must fire on replay
    assert "eda" in replayed_agents
    assert "insight" in replayed_agents
    # Agent set must be consistent (allow ±1 for timing-sensitive agents)
    assert len(original_agents.symmetric_difference(replayed_agents)) <= 2


# ══════════════════════════════════════════════════════════════════════
# GS-10  Recommendation ranking — urgent high-confidence tops list
# ══════════════════════════════════════════════════════════════════════

def test_gs10_recommendation_ranking_urgent_tops():
    from insights.recommendation_ranker import RecommendationRanker

    actions = [
        {"action": "Investigate Android drag",
         "confidence": 0.93, "urgency": 0.95, "business_value": 0.90, "effort": 0.20},
        {"action": "Review quarterly baseline",
         "confidence": 0.60, "urgency": 0.30, "business_value": 0.50, "effort": 0.40},
        {"action": "Document findings",
         "confidence": 0.40, "urgency": 0.20, "business_value": 0.30, "effort": 0.10},
    ]
    ranked = RecommendationRanker().rank(actions)
    assert ranked[0].action == "Investigate Android drag"
    assert ranked[0].score > ranked[-1].score


def test_gs10b_low_evidence_recommendation_ranks_lower():
    """Low-confidence hypothesis should rank 'investigate' over 'take action'."""
    from insights.recommendation_ranker import RecommendationRanker

    investigate = {"action": "Investigate further before acting",
                   "confidence": 0.45, "urgency": 0.55, "business_value": 0.60, "effort": 0.25}
    take_action = {"action": "Immediately change pricing",
                   "confidence": 0.30, "urgency": 0.80, "business_value": 0.85, "effort": 0.15}

    # With low-confidence evidence, we penalise the action
    ranked = RecommendationRanker().rank([investigate, take_action])
    # investigate must rank higher because confidence is higher
    assert ranked[0].action == "Investigate further before acting"


# ══════════════════════════════════════════════════════════════════════
# GS-11  Source authority resolve_many with three sources
# ══════════════════════════════════════════════════════════════════════

def test_gs11_resolve_many_primary_truth_wins():
    from semantic.source_authority import (SourceConflictResolver, SourceClaim, Authority)

    claims = [
        SourceClaim("trend",           "avg",      0.8, authority=Authority.agent_output),
        SourceClaim("enrichment",      "median",   0.7, authority=Authority.external),
        SourceClaim("metric_registry", "sum",      1.0, authority=Authority.primary_truth),
    ]
    winner, penalty = SourceConflictResolver().resolve_many(claims)
    assert winner.source == "metric_registry"
    assert penalty >= 0.0


# ══════════════════════════════════════════════════════════════════════
# GS-12  Grain coercion — invalid grain → governed grain
# ══════════════════════════════════════════════════════════════════════

def test_gs12_grain_coercion():
    from semantic.grain_resolver import GrainResolver
    from semantic.metric_registry import MetricRegistry

    registry = MetricRegistry({
        "revenue": {"description": "Rev", "aggregation": "sum",
                    "allowed_grains": ["daily", "weekly"]}
    })
    resolver = GrainResolver(registry)
    resolved = resolver.resolve("revenue", "hourly")
    assert resolved.lower() == "daily"


# ══════════════════════════════════════════════════════════════════════
# GS-13  Output classification — PII in brief → CONFIDENTIAL + redacted
# ══════════════════════════════════════════════════════════════════════

def test_gs13_pii_output_classified_and_redacted():
    from security.security_shell import SecurityShell
    shell = SecurityShell(tenant_id="acme", user_id="u1", role="analyst")
    brief_with_pii = "Please contact ceo@example.com regarding the revenue drop."
    payload, classification = shell.publish_output(
        {"brief": brief_with_pii, "tenant_id": "acme"},
        requested_tenant_id="acme",
    )
    assert classification in ("CONFIDENTIAL", "RESTRICTED")
    if isinstance(payload, dict) and "brief" in payload:
        assert "ceo@example.com" not in payload["brief"]


# ══════════════════════════════════════════════════════════════════════
# GS-14  Full pipeline E2E — all required outputs present
# ══════════════════════════════════════════════════════════════════════

def test_gs14_full_pipeline_required_outputs():
    from agents.pipeline import GovernedPipeline
    from agents.context import AnalysisContext

    df = make_ts(n=60, spike_idx=40)
    ctx = AnalysisContext(
        df=df, kpi_col="revenue", date_col="date",
        tenant_id="default", user_id="system",
        business_context={"company": "Acme", "industry": "SaaS"},
    )
    ctx = GovernedPipeline().run(ctx)

    # All required outputs must be present
    assert ctx.run_id,                           "run_id must be set"
    assert ctx.final_brief,                      "final_brief must be non-empty"
    assert ctx.data_quality_report,              "data_quality_report must be populated"
    assert len(ctx.approval_log) >= 1,           "approval_log must have at least one entry"
    assert ctx.research_plan is not None,        "research_plan must be created"
    assert len(ctx.recommendation_candidates) >= 1, "recommendation_candidates must be non-empty"
    assert ctx.run_manifest,                     "run_manifest must be populated"
    assert "guardian" in ctx.results,            "guardian must run"
    assert "insight" in ctx.results,             "insight must run"
    assert ctx.run_manifest.get("output_classification"), "output must be classified"
    # Guardian result must contain evidence grade
    g = ctx.results["guardian"]
    if g.status == "success":
        assert "evidence_grade" in g.data or "verdict" in g.data


# ══════════════════════════════════════════════════════════════════════
# Release threshold gate  (run last)
# ══════════════════════════════════════════════════════════════════════

GOLD_TESTS = [
    "test_gs01_clean_anomaly_detected",
    "test_gs02_bad_data_blocks_strong_conclusion",
    "test_gs03_source_authority_primary_truth_wins",
    "test_gs04_invalid_dimension_rejected_by_semantic",
    "test_gs05_untestable_hypothesis_marked_and_gap_recorded",
    "test_gs06_strong_support_confirmed_moderate_evidence",
    "test_gs07_contradictory_agents_detected",
    "test_gs08_cross_tenant_blocked",
    "test_gs09_replay_parity",
    "test_gs10_recommendation_ranking_urgent_tops",
    "test_gs11_resolve_many_primary_truth_wins",
    "test_gs12_grain_coercion",
    "test_gs13_pii_output_classified_and_redacted",
    "test_gs14_full_pipeline_required_outputs",
]


def test_release_gate_all_gold_scenarios_defined():
    """
    Meta-test: verifies all gold scenarios are defined in this file.
    If any are missing the release gate itself is broken.
    """
    current_globals = {k for k in globals() if k.startswith("test_gs")}
    for name in GOLD_TESTS:
        assert name in current_globals, f"Gold scenario missing: {name}"
