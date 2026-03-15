"""
tests/test_bench_source_conflict.py  — v9
Source conflict handling benchmarks (Phase 9).

Covers:
  - Authority lookup for all known source types
  - Two-source resolution: all authority-tier combinations
  - Freshness tiebreaker
  - Completeness tiebreaker
  - Unresolvable conflict → penalty + no winner
  - resolve_many: 3+ sources, winner is highest authority
  - Confidence penalty magnitude checks
  - Integration: conflict in evidence → hypothesis confidence penalised
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from semantic.source_authority import (
    SourceClaim, SourceConflictResolver, Authority, AUTHORITY_MAP
)


# ══════════════════════════════════════════════════════════════════════
# Authority map coverage
# ══════════════════════════════════════════════════════════════════════

class TestAuthorityMap:

    def test_metric_registry_is_primary_truth(self):
        assert AUTHORITY_MAP["metric_registry"] == Authority.primary_truth

    def test_policy_store_is_primary_truth(self):
        assert AUTHORITY_MAP["policy_store"] == Authority.primary_truth

    def test_join_graph_is_primary_truth(self):
        assert AUTHORITY_MAP["join_graph"] == Authority.primary_truth

    def test_connector_is_governed_data(self):
        assert AUTHORITY_MAP["connector"] == Authority.governed_data

    def test_agent_outputs_are_agent_output(self):
        for agent in ("trend", "anomaly", "root_cause", "funnel", "forecast"):
            assert AUTHORITY_MAP[agent] == Authority.agent_output

    def test_enrichment_is_external(self):
        assert AUTHORITY_MAP["enrichment"] == Authority.external

    def test_org_memory_is_prior(self):
        assert AUTHORITY_MAP["org_memory"] == Authority.prior

    def test_unknown_source_gets_unknown_authority(self):
        claim = SourceClaim(source="made_up_source", value="x")
        assert claim.authority == Authority.unknown

    def test_authority_ordering(self):
        assert Authority.primary_truth > Authority.governed_data
        assert Authority.governed_data > Authority.agent_output
        assert Authority.agent_output  > Authority.external
        assert Authority.external      > Authority.prior
        assert Authority.prior         > Authority.unknown


# ══════════════════════════════════════════════════════════════════════
# Two-source resolution
# ══════════════════════════════════════════════════════════════════════

class TestTwoSourceResolution:

    def _claim(self, source, value, freshness="", completeness=1.0):
        return SourceClaim(source=source, value=value,
                           freshness_ts=freshness, completeness=completeness)

    def test_primary_truth_beats_agent(self):
        r = SourceConflictResolver().resolve(
            self._claim("metric_registry", "sum"),
            self._claim("trend",           "avg"),
        )
        assert r.resolved
        assert r.winner.source == "metric_registry"
        assert r.conflict_type == "authority"

    def test_governed_data_beats_external(self):
        r = SourceConflictResolver().resolve(
            self._claim("enrichment", "up"),
            self._claim("connector",  "down"),
        )
        assert r.resolved
        assert r.winner.source == "connector"

    def test_agreement_zero_penalty(self):
        r = SourceConflictResolver().resolve(
            self._claim("trend",   "up"),
            self._claim("anomaly", "up"),
        )
        assert r.resolved
        assert r.confidence_penalty == 0.0

    def test_freshness_tiebreaker(self):
        r = SourceConflictResolver().resolve(
            self._claim("trend",   "up",   freshness="2025-01-01T00:00:00"),
            self._claim("anomaly", "down", freshness="2025-06-01T00:00:00"),
        )
        assert r.resolved
        assert r.winner.source == "anomaly"
        assert r.conflict_type == "freshness"

    def test_completeness_tiebreaker(self):
        r = SourceConflictResolver().resolve(
            self._claim("trend",   "up",   completeness=0.3),
            self._claim("anomaly", "down", completeness=0.95),
        )
        assert r.resolved
        assert r.winner.source == "anomaly"
        assert r.conflict_type == "completeness"

    def test_unresolvable_when_equal_authority_and_no_tiebreaker(self):
        r = SourceConflictResolver().resolve(
            self._claim("trend",   "up"),
            self._claim("anomaly", "down"),
        )
        assert r.resolved is False
        assert r.confidence_penalty >= 0.15
        assert r.conflict_type == "unresolvable"

    def test_penalty_grows_with_authority_gap(self):
        small_gap = SourceConflictResolver().resolve(
            SourceClaim("metric_registry", "A", authority=Authority.primary_truth),
            SourceClaim("connector",       "B", authority=Authority.governed_data),
        )
        large_gap = SourceConflictResolver().resolve(
            SourceClaim("metric_registry", "A", authority=Authority.primary_truth),
            SourceClaim("org_memory",      "B", authority=Authority.prior),
        )
        assert large_gap.confidence_penalty >= small_gap.confidence_penalty


# ══════════════════════════════════════════════════════════════════════
# resolve_many
# ══════════════════════════════════════════════════════════════════════

class TestResolveMany:

    def test_three_sources_highest_authority_wins(self):
        claims = [
            SourceClaim("trend",           "avg", authority=Authority.agent_output),
            SourceClaim("enrichment",      "med", authority=Authority.external),
            SourceClaim("metric_registry", "sum", authority=Authority.primary_truth),
        ]
        winner, penalty = SourceConflictResolver().resolve_many(claims)
        assert winner.source == "metric_registry"

    def test_all_agree_zero_penalty(self):
        claims = [
            SourceClaim("trend",   "up", authority=Authority.agent_output),
            SourceClaim("anomaly", "up", authority=Authority.agent_output),
            SourceClaim("funnel",  "up", authority=Authority.agent_output),
        ]
        winner, penalty = SourceConflictResolver().resolve_many(claims)
        assert penalty == 0.0

    def test_empty_list_returns_none(self):
        winner, penalty = SourceConflictResolver().resolve_many([])
        assert winner is None
        assert penalty == 0.0

    def test_single_claim_no_penalty(self):
        claims = [SourceClaim("trend", "up", authority=Authority.agent_output)]
        winner, penalty = SourceConflictResolver().resolve_many(claims)
        assert winner.source == "trend"
        assert penalty == 0.0

    def test_penalty_capped_at_40_percent(self):
        # Many conflicts should not push penalty > 0.40
        claims = [
            SourceClaim(f"agent_{i}", "different_value_{i}", authority=Authority.agent_output)
            for i in range(10)
        ]
        _, penalty = SourceConflictResolver().resolve_many(claims)
        assert penalty <= 0.40


# ══════════════════════════════════════════════════════════════════════
# Integration: conflict in evidence penalises hypothesis confidence
# ══════════════════════════════════════════════════════════════════════

class TestSourceConflictInHypothesisClosing:

    def test_contradictory_evidence_reduces_confidence(self):
        """
        When two agent results directly contradict each other, the
        ConclusionEngine should produce a lower confidence than when
        both sources agree.
        """
        import uuid
        from agents.context import AnalysisContext, AgentResult
        from science.research_plan import ResearchPlan, Hypothesis, HypothesisStatus
        from science.conclusion_engine import ConclusionEngine
        from tests.conftest import make_ts

        def _run(supports_a, supports_b):
            df = make_ts(n=30)
            ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date")
            ctx.data_quality_report = {"score": 0.85}
            h = Hypothesis(
                id=str(uuid.uuid4()), statement="Rev dropped",
                source="data", status=HypothesisStatus.TESTABLE,
            )
            h.evidence = [
                {"agent": "trend",   "summary": "trend says A", "supports": supports_a, "confidence": 0.85},
                {"agent": "anomaly", "summary": "anomaly says B", "supports": supports_b, "confidence": 0.85},
            ]
            for e in h.evidence:
                ctx.write_result(AgentResult(
                    agent=e["agent"], status="success", summary=e["summary"], data={}))
            ctx.research_plan = ResearchPlan(hypotheses=[h])
            plan = ConclusionEngine().close_hypotheses(ctx)
            return plan.hypotheses[0].confidence

        conf_agreed   = _run(True, True)
        conf_conflict = _run(True, False)

        assert conf_agreed >= conf_conflict, \
            "Agreeing evidence must produce >= confidence vs conflicting evidence"
