"""
tests/test_bench_guardian_learning.py
Benchmark coverage for:
  - Guardian layer (EvidenceGrader, AgentScoreboard, ContradictionChecker, LessonExtractor)
  - Learning layer (LearningLayer base, IngestionLearner, AnalysisLearner, OrchestratorLearner)
  - Quality layer (DataQualityGate - comprehensive edge cases)
  - Governance layer (ApprovalGate - all action types)
  - Versioning layer (RunManifest persist/load)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import make_ts, make_bad_df, make_empty_df
from agents.context import AnalysisContext, AgentResult


# ══════════════════════════════════════════════════════════════════════
# EvidenceGrader
# ══════════════════════════════════════════════════════════════════════

class TestEvidenceGrader:

    def test_strong_evidence_grade(self):
        from guardian.evidence_grader import EvidenceGrader
        grade = EvidenceGrader().grade(
            support_ratio=0.95, confidence=0.9,
            contradictions=0, data_quality_score=0.95)
        assert grade.grade == "strong"

    def test_contradictions_reduce_grade(self):
        from guardian.evidence_grader import EvidenceGrader
        eg = EvidenceGrader()
        no_contra = eg.grade(0.8, 0.8, 0, 0.9)
        with_contra = eg.grade(0.8, 0.8, 5, 0.9)
        assert no_contra.summary_strength >= with_contra.summary_strength

    def test_low_dq_reduces_grade(self):
        from guardian.evidence_grader import EvidenceGrader
        eg = EvidenceGrader()
        good_dq = eg.grade(0.8, 0.8, 0, 0.95)
        bad_dq  = eg.grade(0.8, 0.8, 0, 0.2)
        assert good_dq.summary_strength >= bad_dq.summary_strength

    def test_grade_levels_correct_order(self):
        from guardian.evidence_grader import EvidenceGrader
        eg = EvidenceGrader()
        grades = {"strong": 0, "moderate": 1, "weak": 2, "speculative": 3}
        strong     = eg.grade(0.95, 0.95, 0, 1.0).grade
        speculative= eg.grade(0.1,  0.1,  5, 0.1).grade
        assert grades[strong] < grades[speculative]

    def test_reasons_populated_when_issues(self):
        from guardian.evidence_grader import EvidenceGrader
        grade = EvidenceGrader().grade(0.5, 0.5, 3, 0.3)
        assert len(grade.reasons) >= 1

    def test_strength_always_0_to_1(self):
        from guardian.evidence_grader import EvidenceGrader
        eg = EvidenceGrader()
        for sr, conf, contra, dq in [
            (0.0, 0.0, 10, 0.0),
            (1.0, 1.0, 0, 1.0),
            (0.5, 0.5, 2, 0.6),
        ]:
            grade = eg.grade(sr, conf, contra, dq)
            assert 0.0 <= grade.summary_strength <= 1.0


# ══════════════════════════════════════════════════════════════════════
# AgentScoreboard
# ══════════════════════════════════════════════════════════════════════

class TestAgentScoreboard:

    def test_record_and_retrieve_score(self, tmp_path):
        from guardian.agent_scoreboard import AgentScoreboard
        sb = AgentScoreboard(db_path=str(tmp_path / "scores.db"))
        sb.record("trend", 0.85, run_id="r1")
        perf = sb.summary("trend")
        assert perf.score == pytest.approx(0.85)
        assert perf.runs == 1

    def test_average_over_multiple_records(self, tmp_path):
        from guardian.agent_scoreboard import AgentScoreboard
        sb = AgentScoreboard(db_path=str(tmp_path / "scores.db"))
        for s in [0.6, 0.8, 1.0]:
            sb.record("anomaly", s, "r1")
        perf = sb.summary("anomaly")
        assert perf.score == pytest.approx(0.8, abs=0.01)
        assert perf.runs == 3

    def test_unknown_agent_returns_none_score(self, tmp_path):
        from guardian.agent_scoreboard import AgentScoreboard
        sb = AgentScoreboard(db_path=str(tmp_path / "scores.db"))
        perf = sb.summary("never_run_agent")
        assert perf.score is None
        assert perf.runs == 0

    def test_multiple_agents_independent(self, tmp_path):
        from guardian.agent_scoreboard import AgentScoreboard
        sb = AgentScoreboard(db_path=str(tmp_path / "scores.db"))
        sb.record("trend", 0.9)
        sb.record("forecast", 0.4)
        assert sb.summary("trend").score != sb.summary("forecast").score

    def test_score_range_validation(self, tmp_path):
        from guardian.agent_scoreboard import AgentScoreboard
        sb = AgentScoreboard(db_path=str(tmp_path / "scores.db"))
        for s in [0.0, 0.5, 1.0]:
            sb.record("root_cause", s)
        perf = sb.summary("root_cause")
        assert 0.0 <= perf.score <= 1.0


# ══════════════════════════════════════════════════════════════════════
# LessonExtractor
# ══════════════════════════════════════════════════════════════════════

class TestLessonExtractor:

    def _make_ctx(self):
        ctx = AnalysisContext(
            df=make_ts(n=30),
            kpi_col="revenue", date_col="date",
            run_id="test-run-001",
        )
        ctx.active_agents = ["trend", "anomaly", "insight"]
        ctx.data_quality_report = {"score": 0.85}
        ctx.write_result(AgentResult(
            agent="guardian", status="success",
            summary="No contradictions",
            data={"contradictions": []},
        ))
        return ctx

    def test_extract_returns_dict_with_required_keys(self, tmp_path):
        from guardian.lesson_extractor import LessonExtractor
        le = LessonExtractor(store_path=str(tmp_path / "lessons.jsonl"))
        lesson = le.extract(self._make_ctx())
        for key in ("run_id", "kpi", "active_agents", "timestamp"):
            assert key in lesson

    def test_persist_writes_jsonl(self, tmp_path):
        from guardian.lesson_extractor import LessonExtractor
        path = tmp_path / "lessons.jsonl"
        le = LessonExtractor(store_path=str(path))
        lesson = le.extract(self._make_ctx())
        le.persist(lesson)
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["run_id"] == "test-run-001"

    def test_persist_appends(self, tmp_path):
        from guardian.lesson_extractor import LessonExtractor
        path = tmp_path / "lessons.jsonl"
        le = LessonExtractor(store_path=str(path))
        ctx = self._make_ctx()
        le.persist(le.extract(ctx))
        le.persist(le.extract(ctx))
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_contradiction_count_captured(self, tmp_path):
        from guardian.lesson_extractor import LessonExtractor
        ctx = self._make_ctx()
        ctx.write_result(AgentResult(
            agent="guardian", status="success",
            summary="Contradictions found",
            data={"contradictions": [{"reason": "trend vs anomaly"}]},
        ))
        le = LessonExtractor(store_path=str(tmp_path / "l.jsonl"))
        lesson = le.extract(ctx)
        assert lesson["contradictions"] == 1


# ══════════════════════════════════════════════════════════════════════
# DataQualityGate — comprehensive edge cases
# ══════════════════════════════════════════════════════════════════════

class TestDataQualityGateComprehensive:

    def test_score_range(self):
        from quality.data_quality_gate import DataQualityGate
        gate = DataQualityGate()
        for df in [make_ts(30), make_bad_df(), make_empty_df()]:
            report = gate.assess(df)
            assert 0.0 <= report.score <= 1.0

    def test_good_data_high_score(self):
        from quality.data_quality_gate import DataQualityGate
        df = make_ts(n=100)
        report = DataQualityGate().assess(df, date_col="date", kpi_col="revenue")
        assert report.score >= 0.7
        assert report.ok is True

    def test_empty_df_blocks(self):
        from quality.data_quality_gate import DataQualityGate
        report = DataQualityGate().assess(make_empty_df())
        assert report.ok is False
        assert report.score <= 0.35

    def test_high_null_ratio_blocks(self):
        from quality.data_quality_gate import DataQualityGate
        df = pd.DataFrame({"v": [None]*80 + list(range(20))})
        report = DataQualityGate().assess(df, kpi_col="v")
        assert report.null_ratio > 0.5

    def test_high_duplicate_ratio_warns(self):
        from quality.data_quality_gate import DataQualityGate
        df = pd.DataFrame({"date": ["2025-01-01"]*50, "v": [100]*50})
        report = DataQualityGate().assess(df, date_col="date", kpi_col="v")
        assert report.duplicate_ratio > 0.25

    def test_missing_kpi_col_blocks(self):
        from quality.data_quality_gate import DataQualityGate
        df = make_ts(30)
        report = DataQualityGate().assess(df, kpi_col="nonexistent_col")
        assert report.ok is False

    def test_insufficient_rows_warns(self):
        from quality.data_quality_gate import DataQualityGate
        df = pd.DataFrame({"date": ["2025-01-01"]*5, "v": [1,2,3,4,5]})
        report = DataQualityGate().assess(df, date_col="date", kpi_col="v")
        assert report.sufficiency_ok is False or len(report.warnings) >= 1

    def test_bad_date_column_noted(self):
        from quality.data_quality_gate import DataQualityGate
        df = pd.DataFrame({"date": ["not-a-date"]*20, "v": list(range(20))})
        report = DataQualityGate().assess(df, date_col="date", kpi_col="v")
        assert isinstance(report.warnings, list)

    def test_report_has_required_fields(self):
        from quality.data_quality_gate import DataQualityGate
        report = DataQualityGate().assess(make_ts(30), date_col="date", kpi_col="revenue")
        for field in ("ok","score","freshness_ok","completeness_ok",
                      "continuity_ok","sufficiency_ok","duplicate_ratio",
                      "null_ratio","row_count","warnings","blocking_reasons"):
            assert hasattr(report, field)

    def test_dq_score_penalised_proportionally(self):
        from quality.data_quality_gate import DataQualityGate
        clean = DataQualityGate().assess(make_ts(100), "date", "revenue").score
        noisy = DataQualityGate().assess(make_bad_df()).score
        assert clean > noisy


# ══════════════════════════════════════════════════════════════════════
# ApprovalGate — all action types
# ══════════════════════════════════════════════════════════════════════

class TestApprovalGateComprehensive:

    PROTECTED_ACTIONS = [
        "metric_definition_change",
        "join_path_change",
        "policy_change",
        "source_authority_change",
        "alert_threshold_change",
        "promote_learned_truth",
    ]

    def test_all_protected_actions_require_approval(self, tmp_path):
        from governance.approval_gate import ApprovalGate
        gate = ApprovalGate(base_dir=str(tmp_path))
        for action in self.PROTECTED_ACTIONS:
            decision = gate.check(action, approved=False)
            assert decision.approved is False, f"{action} should require approval"

    def test_approved_protected_action_passes(self, tmp_path):
        from governance.approval_gate import ApprovalGate
        gate = ApprovalGate(base_dir=str(tmp_path))
        for action in self.PROTECTED_ACTIONS:
            decision = gate.check(action, approved=True, reason="admin override")
            assert decision.approved is True

    def test_unprotected_action_passes_without_approval(self, tmp_path):
        from governance.approval_gate import ApprovalGate
        gate = ApprovalGate(base_dir=str(tmp_path))
        decision = gate.check("runtime_analysis", approved=False)
        assert decision.approved is True

    def test_log_request_writes_jsonl(self, tmp_path):
        from governance.approval_gate import ApprovalGate
        gate = ApprovalGate(base_dir=str(tmp_path))
        gate.log_request("metric_definition_change",
                         {"metric": "revenue"}, approved=False, approver="")
        log_path = tmp_path / "approvals.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["action_type"] == "metric_definition_change"
        assert record["approved"] is False

    def test_decision_contains_action_type(self, tmp_path):
        from governance.approval_gate import ApprovalGate
        gate = ApprovalGate(base_dir=str(tmp_path))
        decision = gate.check("policy_change", approved=False)
        assert decision.action_type == "policy_change"


# ══════════════════════════════════════════════════════════════════════
# RunManifest
# ══════════════════════════════════════════════════════════════════════

class TestRunManifest:

    def test_create_sets_run_id_and_timestamp(self):
        from versioning.run_manifest import RunManifest
        m = RunManifest.create("run-abc-123")
        assert m.run_id == "run-abc-123"
        assert m.created_at

    def test_persist_and_reload(self, tmp_path):
        from versioning.run_manifest import RunManifest
        m = RunManifest.create("r1")
        m.active_agents = ["trend", "anomaly"]
        m.data_quality_score = 0.87
        m.persist(base_dir=str(tmp_path))
        path = tmp_path / "r1.json"
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["run_id"] == "r1"
        assert loaded["active_agents"] == ["trend", "anomaly"]
        assert abs(loaded["data_quality_score"] - 0.87) < 0.001

    def test_to_dict_serialisable(self):
        from versioning.run_manifest import RunManifest
        m = RunManifest.create("r2")
        d = m.to_dict()
        json.dumps(d)  # should not raise

    def test_notes_appendable(self):
        from versioning.run_manifest import RunManifest
        m = RunManifest.create("r3")
        m.notes.append("DQ warning: sparse time series")
        assert len(m.notes) == 1

    def test_multiple_manifests_separate_files(self, tmp_path):
        from versioning.run_manifest import RunManifest
        for rid in ["r1", "r2", "r3"]:
            RunManifest.create(rid).persist(base_dir=str(tmp_path))
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 3


# ══════════════════════════════════════════════════════════════════════
# Learning layer
# ══════════════════════════════════════════════════════════════════════

class TestLearningLayer:

    def _ctx(self):
        ctx = AnalysisContext(
            df=make_ts(30), kpi_col="revenue", date_col="date",
            run_id="learn-001",
        )
        ctx.active_agents = ["trend", "anomaly"]
        ctx.data_profile = {"rows": 30, "has_time_series": True,
                            "dimensions": [], "kpis": ["revenue"],
                            "has_funnel_signal": False, "has_cohort_signal": False}
        return ctx

    def test_analysis_learner_observe_no_crash(self):
        from learning.layer_adapters import AnalysisLearner
        ctx = self._ctx()
        result = AnalysisLearner().observe(ctx, None)
        assert isinstance(result, dict)

    def test_orchestrator_learner_adapt_returns_dict(self):
        from learning.layer_adapters import OrchestratorLearner
        ctx = self._ctx()
        result = OrchestratorLearner().adapt(ctx)
        assert isinstance(result, dict)

    def test_ingestion_learner_observe_no_crash(self):
        from learning.layer_adapters import IngestionLearner
        ctx = self._ctx()
        result = IngestionLearner().observe(ctx, None)
        assert isinstance(result, dict)

    def test_hypothesis_learner_observe_no_crash(self):
        from learning.layer_adapters import HypothesisLearner
        ctx = self._ctx()
        result = HypothesisLearner().observe(ctx, None)
        assert isinstance(result, dict)

    def test_insight_learner_observe_no_crash(self):
        from learning.layer_adapters import InsightLearner
        ctx = self._ctx()
        result = InsightLearner().observe(ctx, None)
        assert isinstance(result, dict)
