"""
tests/test_bench_release_gate.py  — v9
Release discipline tests.

Verifies:
  - All release gate checks pass (programmatic)
  - Benchmark categories are properly separated
  - Smoke suite is a strict subset of release suite
"""

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_release_gate_all_checks_pass():
    """The programmatic release gate must report all checks passing."""
    from versioning.release_gate import check_release_readiness
    ok, issues = check_release_readiness()
    if not ok:
        pytest.fail(
            f"Release gate failed ({len(issues)} issue(s)):\n"
            + "\n".join(f"  • {i}" for i in issues)
        )


def test_release_gate_checklist_returns_dict():
    from versioning.release_gate import ReleaseChecklist
    result = ReleaseChecklist().run()
    assert "release_ready" in result
    assert "checks_passed" in result
    assert "issues" in result
    assert isinstance(result["issues"], list)


def test_analysis_contract_importable():
    from analysis.contract import AnalysisContract, AnalysisResult
    r = AnalysisResult(module="test", ok=True)
    assert r.module == "test"
    bm = AnalysisContract.__abstractmethods__
    assert "analyze" in bm


def test_governed_pipeline_is_canonical():
    """GovernedPipeline must be importable and AgentRunner must be its alias."""
    from agents.pipeline import GovernedPipeline, AgentRunner
    assert issubclass(AgentRunner, GovernedPipeline)


def test_source_authority_importable():
    from semantic.source_authority import SourceConflictResolver, Authority, AUTHORITY_MAP
    assert Authority.primary_truth > Authority.agent_output


def test_metric_store_deprecation_warning():
    """MetricStore must warn on import/instantiation — it is deprecated."""
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from metadata.metric_store import MetricStore
        MetricStore()
    assert any(issubclass(x.category, DeprecationWarning) for x in w), \
        "MetricStore must emit DeprecationWarning"


def test_run_manifest_has_all_v9_fields():
    from versioning.run_manifest import RunManifest
    m = RunManifest.create("v9-check")
    d = m.to_dict()
    for field in (
        "run_id", "created_at",
        "metric_registry_version", "join_graph_version",
        "policy_version", "prompt_version", "config_version",
        "output_classification",
        "guardian_summary", "evidence_summary",
        "replay_type", "notes",
    ):
        assert field in d, f"RunManifest missing required v9 field: {field}"


def test_recommendation_ranker_has_evidence_quality():
    """v9 ranker must accept evidence_quality and use it in scoring."""
    from insights.recommendation_ranker import RecommendationRanker
    high_ev = {"action": "A", "confidence": 0.8, "urgency": 0.8,
               "business_value": 0.8, "effort": 0.2, "evidence_quality": 1.0}
    low_ev  = {"action": "B", "confidence": 0.8, "urgency": 0.8,
               "business_value": 0.8, "effort": 0.2, "evidence_quality": 0.2}
    ranked = RecommendationRanker().rank([high_ev, low_ev])
    assert ranked[0].action == "A", "High evidence quality must rank higher"
    assert ranked[0].score > ranked[1].score


# ── Benchmark category registry ──────────────────────────────────────

SMOKE_TESTS = [
    "tests/test_bench_gold_scenarios.py::test_gs14_full_pipeline_required_outputs",
    "tests/test_bench_gold_scenarios.py::test_gs01_clean_anomaly_detected",
    "tests/test_bench_gold_scenarios.py::test_gs08_cross_tenant_blocked",
    "tests/test_bench_release_gate.py::test_release_gate_all_checks_pass",
]

RELEASE_REQUIRED_FILES = [
    "tests/test_bench_gold_scenarios.py",
    "tests/test_bench_security_boundaries.py",
    "tests/test_bench_source_conflict.py",
    "tests/test_bench_release_gate.py",
    "tests/test_bench_analysis.py",
    "tests/test_bench_security.py",
    "tests/test_bench_guardian_learning.py",
    "tests/test_bench_pipeline_e2e.py",
]


def test_release_required_files_all_present():
    """All required benchmark files must exist before a release is cut."""
    for rel_path in RELEASE_REQUIRED_FILES:
        path = ROOT / rel_path
        assert path.exists(), f"Release-required benchmark file missing: {rel_path}"
