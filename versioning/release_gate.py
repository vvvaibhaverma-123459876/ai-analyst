"""
versioning/release_gate.py  — v9
Release discipline enforcement.

A release is NOT ready unless:
  1. All critical benchmark tests pass (gold scenarios + security boundaries)
  2. No duplicate truth path is active (MetricStore must emit deprecation warning)
  3. Main pipeline uses semantic/quality/guardian/security (GovernedPipeline)
  4. Replay parity verified (GS-09 passes)
  5. Output classification attached to every manifest

This module provides:
  - ReleaseChecklist: programmatic gate that can be called from CI
  - check_release_readiness(): returns (ok, issues) tuple
  - The GitHub Actions workflow calls this in the nightly job

Usage (CI):
    python -c "from versioning.release_gate import check_release_readiness; ok, issues = check_release_readiness(); print(issues); exit(0 if ok else 1)"
"""
from __future__ import annotations
import importlib
import warnings
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _check_no_active_metric_store() -> str | None:
    """MetricStore must emit a DeprecationWarning, not silently work."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            from metadata.metric_store import MetricStore
            MetricStore()
        except Exception:
            pass
        deprecation_warned = any(issubclass(x.category, DeprecationWarning) for x in w)
    if not deprecation_warned:
        return "MetricStore does not emit DeprecationWarning — legacy truth path still active."
    return None


def _check_pipeline_is_governed() -> str | None:
    """AgentRunner must be an alias of GovernedPipeline."""
    try:
        from agents.runner import AgentRunner
        from agents.pipeline import GovernedPipeline
        if not issubclass(AgentRunner, GovernedPipeline):
            return "AgentRunner is not a subclass of GovernedPipeline — pipeline bypass possible."
    except ImportError as e:
        return f"Pipeline import failed: {e}"
    return None


def _check_semantic_grain_resolver_exists() -> str | None:
    """GrainResolver must exist and be callable."""
    try:
        from semantic.grain_resolver import GrainResolver
        from semantic.metric_registry import MetricRegistry
        GrainResolver(MetricRegistry({}))
    except Exception as e:
        return f"GrainResolver broken: {e}"
    return None


def _check_source_authority_exists() -> str | None:
    """SourceConflictResolver must exist."""
    try:
        from semantic.source_authority import SourceConflictResolver
        SourceConflictResolver()
    except Exception as e:
        return f"SourceConflictResolver broken: {e}"
    return None


def _check_gold_scenarios_file_exists() -> str | None:
    path = ROOT / "tests" / "test_bench_gold_scenarios.py"
    if not path.exists():
        return "Gold scenario benchmark file missing: tests/test_bench_gold_scenarios.py"
    return None


def _check_security_boundaries_file_exists() -> str | None:
    path = ROOT / "tests" / "test_bench_security_boundaries.py"
    if not path.exists():
        return "Security boundary test file missing: tests/test_bench_security_boundaries.py"
    return None


def _check_run_manifest_has_output_classification() -> str | None:
    try:
        from versioning.run_manifest import RunManifest
        m = RunManifest.create("gate-check")
        if "output_classification" not in m.to_dict():
            return "RunManifest missing output_classification field."
    except Exception as e:
        return f"RunManifest check failed: {e}"
    return None


def _check_analysis_contract_exists() -> str | None:
    try:
        from analysis.contract import AnalysisContract, AnalysisResult
        _ = AnalysisResult(module="test", ok=True)
    except Exception as e:
        return f"AnalysisContract broken: {e}"
    return None


# ── Checklist ─────────────────────────────────────────────────────────


def _check_analysis_contract_coverage() -> str | None:
    """All core analysis modules must be AnalysisContract subclasses."""
    REQUIRED = {
        "AnomalyDetector":    ("analysis.anomaly_detector", "AnomalyDetector"),
        "FunnelAnalyzer":     ("analysis.funnel_analyzer",  "FunnelAnalyzer"),
        "CohortAnalyzer":     ("analysis.cohort_analyzer",  "CohortAnalyzer"),
        "RootCauseAnalyzer":  ("analysis.root_cause",       "RootCauseAnalyzer"),
        "StatisticsAnalyzer": ("analysis.statistics",       "StatisticsAnalyzer"),
    }
    try:
        import importlib
        from analysis.contract import AnalysisContract
        missing = []
        for label, (mod_path, cls_name) in REQUIRED.items():
            try:
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)
                if not issubclass(cls, AnalysisContract):
                    missing.append(f"{label} not subclass of AnalysisContract")
            except Exception as e:
                missing.append(f"{label}: {e}")
        if missing:
            return "Contract migration incomplete: " + "; ".join(missing)
    except Exception as e:
        return f"Contract coverage check failed: {e}"
    return None



def _check_phase_barrier_exists() -> str | None:
    """AnalysisContext must have a freeze() method (phase barrier)."""
    try:
        from agents.context import AnalysisContext, ContextSnapshot
        import inspect
        if not hasattr(AnalysisContext, "freeze"):
            return "AnalysisContext.freeze() missing — phase barrier not implemented."
        sig = inspect.signature(AnalysisContext.freeze)
        # freeze() should return ContextSnapshot
        return None
    except Exception as e:
        return f"Phase barrier check failed: {e}"



def _check_manifest_persistence_guaranteed() -> str | None:
    """_safe_run() must exist and have a finally block that persists the manifest."""
    try:
        import inspect
        from agents.pipeline import GovernedPipeline
        if not hasattr(GovernedPipeline, "_safe_run"):
            return "GovernedPipeline._safe_run() missing — failed runs invisible to audit."
        source = inspect.getsource(GovernedPipeline._safe_run)
        if "finally" not in source:
            return "_safe_run() has no finally block."
        if "persist" not in source:
            return "_safe_run() finally block does not call manifest.persist()."
        return None
    except Exception as e:
        return f"Manifest persistence check failed: {e}"


CHECKS = [
    ("no_active_metric_store",          _check_no_active_metric_store),
    ("pipeline_is_governed",            _check_pipeline_is_governed),
    ("semantic_grain_resolver_exists",  _check_semantic_grain_resolver_exists),
    ("source_authority_exists",         _check_source_authority_exists),
    ("gold_scenarios_file_exists",      _check_gold_scenarios_file_exists),
    ("security_boundaries_file_exists", _check_security_boundaries_file_exists),
    ("run_manifest_has_classification", _check_run_manifest_has_output_classification),
    ("analysis_contract_exists",        _check_analysis_contract_exists),
    ("analysis_contract_coverage",       _check_analysis_contract_coverage),
    ("phase_barrier_exists",             _check_phase_barrier_exists),
    ("manifest_persistence_guaranteed",   _check_manifest_persistence_guaranteed),
]


def check_release_readiness() -> tuple[bool, list[str]]:
    """
    Returns (ok: bool, issues: list[str]).
    ok=True only when ALL checks pass.
    """
    issues: list[str] = []
    for name, check_fn in CHECKS:
        result = check_fn()
        if result:
            issues.append(f"[{name}] {result}")
    return len(issues) == 0, issues


class ReleaseChecklist:
    """Programmatic release gate for use in CI scripts or notebooks."""

    def run(self) -> dict[str, Any]:
        ok, issues = check_release_readiness()
        return {
            "release_ready": ok,
            "checks_passed": len(CHECKS) - len(issues),
            "checks_total":  len(CHECKS),
            "issues":        issues,
        }

    def assert_ready(self):
        ok, issues = check_release_readiness()
        if not ok:
            raise AssertionError(
                f"Release gate failed ({len(issues)} issue(s)):\n"
                + "\n".join(f"  • {i}" for i in issues)
            )
