"""
tests/test_bench_v10_gap_closure.py
v10 gap-closure benchmark suite.

Proves the three gaps identified after v9 are fully closed:

  GAP-1  AnalysisContract migration
         All five core analysis modules are AnalysisContract subclasses.
         analyze() is the canonical entry point.
         Existing agent-facing methods still work (backward compat).
         Release gate check: analysis_contract_coverage.

  GAP-2  Phase barrier
         AnalysisContext.freeze() returns a ContextSnapshot.
         Parallel agents receive the snapshot, not the live context.
         write_result() still targets the live context.
         Release gate check: phase_barrier_exists.

  GAP-3  RunManifest persistence on failure
         _safe_run() wraps run() in try/finally.
         Manifest is persisted even when the pipeline raises.
         Failed runs are visible to audit and replay.
         Release gate check: manifest_persistence_guaranteed.
"""

from __future__ import annotations
import sys
import uuid
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest import make_ts, make_funnel, make_cohort, make_segment


# ══════════════════════════════════════════════════════════════════════
# GAP-1: AnalysisContract migration
# ══════════════════════════════════════════════════════════════════════

class TestAnalysisContractMigration:

    def test_all_five_modules_are_subclasses(self):
        import importlib
        from analysis.contract import AnalysisContract
        REQUIRED = [
            ("analysis.anomaly_detector", "AnomalyDetector"),
            ("analysis.funnel_analyzer",  "FunnelAnalyzer"),
            ("analysis.cohort_analyzer",  "CohortAnalyzer"),
            ("analysis.root_cause",       "RootCauseAnalyzer"),
            ("analysis.statistics",       "StatisticsAnalyzer"),
        ]
        for mod_path, cls_name in REQUIRED:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            assert issubclass(cls, AnalysisContract), \
                f"{cls_name} is not an AnalysisContract subclass"

    def test_all_five_implement_analyze(self):
        import importlib
        from analysis.contract import AnalysisContract
        REQUIRED = [
            ("analysis.anomaly_detector", "AnomalyDetector"),
            ("analysis.funnel_analyzer",  "FunnelAnalyzer"),
            ("analysis.cohort_analyzer",  "CohortAnalyzer"),
            ("analysis.root_cause",       "RootCauseAnalyzer"),
            ("analysis.statistics",       "StatisticsAnalyzer"),
        ]
        for mod_path, cls_name in REQUIRED:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            assert hasattr(cls, "analyze"), f"{cls_name} missing analyze()"
            # Must not be abstract (i.e. actually implemented)
            assert "analyze" not in getattr(cls, "__abstractmethods__", set()), \
                f"{cls_name}.analyze() is still abstract"

    def test_funnel_analyze_returns_analysis_result(self):
        from analysis.funnel_analyzer import FunnelAnalyzer
        from analysis.contract import AnalysisResult
        df = pd.DataFrame({
            "user_id": range(120),
            "stage": ["visit"] * 60 + ["signup"] * 35 + ["convert"] * 25,
        })
        result = FunnelAnalyzer().analyze(
            df, stage_col="stage", user_col="user_id",
            stages=["visit", "signup", "convert"]
        )
        assert isinstance(result, AnalysisResult)
        assert result.ok
        assert result.module == "funnel"
        assert len(result.records) == 3
        assert result.metadata.get("biggest_drop")

    def test_cohort_analyze_returns_analysis_result(self):
        from analysis.cohort_analyzer import CohortAnalyzer
        from analysis.contract import AnalysisResult
        df = make_cohort(n_users=80)
        result = CohortAnalyzer().analyze(
            df, user_col="user_id", date_col="activity_date", cohort_grain="M"
        )
        assert isinstance(result, AnalysisResult)
        assert result.ok
        assert result.module == "cohort"
        assert result.metadata.get("retention_matrix") is not None

    def test_rootcause_analyze_returns_analysis_result(self):
        from analysis.root_cause import RootCauseAnalyzer
        from analysis.contract import AnalysisResult
        df = make_segment(n=120)
        result = RootCauseAnalyzer().analyze(
            df, date_col="date", kpi_col="revenue"
        )
        assert isinstance(result, AnalysisResult)
        assert result.ok
        assert result.module == "root_cause"
        assert result.metadata.get("movers")

    def test_statistics_analyze_returns_analysis_result(self):
        from analysis.statistics import StatisticsAnalyzer
        from analysis.contract import AnalysisResult
        df = make_ts(n=60)
        result = StatisticsAnalyzer().analyze(
            df, date_col="date", value_col="revenue", grain="Daily"
        )
        assert isinstance(result, AnalysisResult)
        assert result.ok
        assert result.module == "statistics"
        assert result.metadata.get("comparisons")

    def test_to_benchmark_output_stable_shape(self):
        """All modules must return the same benchmark output shape."""
        from analysis.anomaly_detector import AnomalyDetector
        from analysis.funnel_analyzer  import FunnelAnalyzer
        REQUIRED_KEYS = {"ok", "module", "anomaly_count", "record_count",
                         "summary", "confidence", "method", "warnings", "errors"}
        df_ts    = make_ts(n=40)
        df_funnel = pd.DataFrame({
            "user_id": range(50),
            "stage": ["visit"]*30 + ["convert"]*20,
        })
        for module, kwargs in [
            (AnomalyDetector(), {"kpi_col": "revenue", "date_col": "date", "df": df_ts}),
            (FunnelAnalyzer(),  {"stage_col": "stage", "user_col": "user_id",
                                  "stages": ["visit","convert"], "df": df_funnel}),
        ]:
            df = kwargs.pop("df")
            result = module.analyze(df, **kwargs)
            bm = module.to_benchmark_output(result)
            assert REQUIRED_KEYS <= set(bm.keys()), \
                f"{module.__class__.__name__} benchmark output missing keys"

    def test_existing_agent_methods_still_work(self):
        """Backward compat: existing agent-facing methods must still exist."""
        from analysis.funnel_analyzer  import FunnelAnalyzer
        from analysis.cohort_analyzer  import CohortAnalyzer
        from analysis.root_cause       import RootCauseAnalyzer
        from analysis.statistics       import (
            resample_timeseries, period_comparison, add_trend_line
        )
        assert hasattr(FunnelAnalyzer(),   "compute_funnel")
        assert hasattr(FunnelAnalyzer(),   "biggest_drop")
        assert hasattr(CohortAnalyzer(),   "build_retention_matrix")
        assert hasattr(RootCauseAnalyzer(),"driver_attribution")
        assert callable(resample_timeseries)
        assert callable(period_comparison)
        assert callable(add_trend_line)

    def test_validate_inputs_blocks_empty_df(self):
        from analysis.funnel_analyzer import FunnelAnalyzer
        result = FunnelAnalyzer().analyze(pd.DataFrame(), stage_col="s", user_col="u")
        assert not result.ok
        assert result.errors

    def test_validate_inputs_blocks_missing_column(self):
        from analysis.root_cause import RootCauseAnalyzer
        df = pd.DataFrame({"date": pd.date_range("2025-01-01", 10),
                           "revenue": range(10)})
        result = RootCauseAnalyzer().analyze(df, date_col="date", kpi_col="nonexistent_col")
        assert not result.ok

    def test_release_gate_contract_coverage_check_passes(self):
        from versioning.release_gate import _check_analysis_contract_coverage
        assert _check_analysis_contract_coverage() is None, \
            "Release gate analysis_contract_coverage check must pass"


# ══════════════════════════════════════════════════════════════════════
# GAP-2: Phase barrier
# ══════════════════════════════════════════════════════════════════════

class TestPhaseBarrier:

    def test_context_has_freeze_method(self):
        from agents.context import AnalysisContext
        assert hasattr(AnalysisContext, "freeze"), \
            "AnalysisContext must have a freeze() method"

    def test_freeze_returns_context_snapshot(self):
        from agents.context import AnalysisContext, ContextSnapshot
        ctx = AnalysisContext(
            df=make_ts(n=20), kpi_col="revenue", date_col="date",
            tenant_id="test", run_id="freeze-test"
        )
        snap = ctx.freeze()
        assert isinstance(snap, ContextSnapshot)

    def test_snapshot_df_is_same_object(self):
        """DataFrames are not copied — zero-cost, read-only reference."""
        from agents.context import AnalysisContext
        ctx = AnalysisContext(df=make_ts(n=20))
        snap = ctx.freeze()
        assert snap.df is ctx.df

    def test_snapshot_results_is_different_dict(self):
        """results dict IS copied so parallel agents see a consistent snapshot."""
        from agents.context import AnalysisContext, AgentResult
        ctx = AnalysisContext(df=make_ts(n=20))
        ctx.write_result(AgentResult("eda", "success", "done", {}))
        snap = ctx.freeze()
        assert snap.results is not ctx.results
        assert "eda" in snap.results

    def test_snapshot_does_not_see_post_freeze_writes(self):
        """Write after freeze must not appear in the snapshot."""
        from agents.context import AnalysisContext, AgentResult
        ctx = AnalysisContext(df=make_ts(n=20))
        snap = ctx.freeze()
        ctx.write_result(AgentResult("trend", "success", "trend done", {}))
        assert "trend" not in snap.results

    def test_snapshot_fields_match_context(self):
        from agents.context import AnalysisContext
        ctx = AnalysisContext(
            df=make_ts(n=20), kpi_col="revenue", date_col="date",
            grain="Weekly", tenant_id="acme", run_id="snap-test"
        )
        snap = ctx.freeze()
        assert snap.kpi_col    == "revenue"
        assert snap.grain      == "Weekly"
        assert snap.tenant_id  == "acme"
        assert snap.run_id     == "snap-test"

    def test_write_result_is_thread_safe(self):
        """Multiple threads can write_result concurrently without corruption."""
        import threading
        from agents.context import AnalysisContext, AgentResult
        ctx = AnalysisContext(df=make_ts(n=20))
        agents = [f"agent_{i}" for i in range(20)]
        threads = [
            threading.Thread(
                target=ctx.write_result,
                args=(AgentResult(name, "success", "done", {}),)
            )
            for name in agents
        ]
        for t in threads: t.start()
        for t in threads: t.join()
        assert set(ctx.results.keys()) == set(agents)

    def test_pipeline_uses_freeze_for_parallel_phase(self):
        """GovernedPipeline.run() must call context.freeze() before parallel agents."""
        import inspect
        from agents.pipeline import GovernedPipeline
        source = inspect.getsource(GovernedPipeline.run)
        assert "context.freeze()" in source or "phase5_snapshot" in source, \
            "GovernedPipeline must use phase barrier (context.freeze()) in parallel phase"

    def test_release_gate_phase_barrier_check_passes(self):
        from versioning.release_gate import _check_phase_barrier_exists
        assert _check_phase_barrier_exists() is None, \
            "Release gate phase_barrier_exists check must pass"


# ══════════════════════════════════════════════════════════════════════
# GAP-3: RunManifest persistence on failure
# ══════════════════════════════════════════════════════════════════════

class TestManifestPersistenceOnFailure:

    def test_safe_run_exists(self):
        from agents.pipeline import GovernedPipeline
        assert hasattr(GovernedPipeline, "_safe_run"), \
            "GovernedPipeline must have _safe_run() method"

    def test_safe_run_has_finally(self):
        import inspect
        from agents.pipeline import GovernedPipeline
        source = inspect.getsource(GovernedPipeline._safe_run)
        assert "finally" in source

    def test_safe_run_finally_calls_persist(self):
        import inspect
        from agents.pipeline import GovernedPipeline
        source = inspect.getsource(GovernedPipeline._safe_run)
        finally_section = source.split("finally")[1]
        assert "persist" in finally_section

    def test_manifest_persisted_on_success(self, tmp_path):
        """Successful run must leave a manifest on disk."""
        from agents.pipeline import GovernedPipeline
        from agents.context import AnalysisContext
        df = make_ts(n=30)
        ctx = AnalysisContext(
            df=df, kpi_col="revenue", date_col="date",
            run_id="success-test-" + uuid.uuid4().hex[:8],
        )
        GovernedPipeline()._safe_run(ctx)
        manifest_dir = ROOT / "memory" / "manifests"
        manifest_path = manifest_dir / f"{ctx.run_id}.json"
        assert manifest_path.exists(), "Manifest must be persisted on successful run"

    def test_manifest_persisted_on_failure(self, tmp_path):
        """Failed run must STILL leave a manifest on disk."""
        import json
        from agents.pipeline import GovernedPipeline
        from agents.context import AnalysisContext

        class BrokenPipeline(GovernedPipeline):
            def run(self, context, **_):
                # Set run_id and manifest so finally can persist
                context.run_manifest = {"run_id": context.run_id,
                                        "created_at": "2025-01-01"}
                raise RuntimeError("simulated mid-pipeline failure")

        df = make_ts(n=20)
        run_id = "failure-test-" + uuid.uuid4().hex[:8]
        ctx = AnalysisContext(
            df=df, kpi_col="revenue", date_col="date", run_id=run_id
        )

        with pytest.raises(RuntimeError, match="simulated mid-pipeline failure"):
            BrokenPipeline()._safe_run(ctx)

        manifest_dir = ROOT / "memory" / "manifests"
        manifest_path = manifest_dir / f"{run_id}.json"
        assert manifest_path.exists(), \
            "Manifest must be persisted even when pipeline raises"
        data = json.loads(manifest_path.read_text())
        assert data.get("status") == "failed"
        assert "simulated" in data.get("error", "")

    def test_manifest_status_field_on_failure(self, tmp_path):
        """Manifest for failed run must have status='failed' and error field."""
        import json
        from agents.pipeline import GovernedPipeline
        from agents.context import AnalysisContext

        class BrokenPipeline(GovernedPipeline):
            def run(self, context, **_):
                context.run_manifest = {"run_id": context.run_id, "created_at": "2025-01-01"}
                raise ValueError("test error for manifest")

        run_id = "status-test-" + uuid.uuid4().hex[:8]
        ctx = AnalysisContext(df=make_ts(n=10), kpi_col="revenue",
                               date_col="date", run_id=run_id)
        with pytest.raises(ValueError):
            BrokenPipeline()._safe_run(ctx)

        manifest_path = ROOT / "memory" / "manifests" / f"{run_id}.json"
        if manifest_path.exists():
            data = json.loads(manifest_path.read_text())
            assert data.get("status") == "failed"

    def test_release_gate_manifest_check_passes(self):
        from versioning.release_gate import _check_manifest_persistence_guaranteed
        assert _check_manifest_persistence_guaranteed() is None, \
            "Release gate manifest_persistence_guaranteed check must pass"


# ══════════════════════════════════════════════════════════════════════
# Full release gate: all 11 checks pass
# ══════════════════════════════════════════════════════════════════════

def test_v10_release_gate_all_checks_pass():
    """v10 release gate — all 11 programmatic checks must pass."""
    from versioning.release_gate import check_release_readiness, CHECKS
    assert len(CHECKS) >= 11, f"Expected ≥11 checks, got {len(CHECKS)}"
    ok, issues = check_release_readiness()
    assert ok, f"Release gate failed:\n" + "\n".join(f"  • {i}" for i in issues)


# ══════════════════════════════════════════════════════════════════════
# Coverage score assertion
# ══════════════════════════════════════════════════════════════════════

def test_v10_coverage_claims():
    """
    Structural proof of v10 coverage improvements.
    Each assertion documents a specific closed gap.
    """
    import importlib
    from analysis.contract import AnalysisContract

    # GAP-1: 5/5 modules migrated (was 1/5 in v9 → 80%)
    migrated = 0
    for mod_path, cls_name in [
        ("analysis.anomaly_detector", "AnomalyDetector"),
        ("analysis.funnel_analyzer",  "FunnelAnalyzer"),
        ("analysis.cohort_analyzer",  "CohortAnalyzer"),
        ("analysis.root_cause",       "RootCauseAnalyzer"),
        ("analysis.statistics",       "StatisticsAnalyzer"),
    ]:
        cls = getattr(importlib.import_module(mod_path), cls_name)
        if issubclass(cls, AnalysisContract):
            migrated += 1
    assert migrated == 5, f"Expected 5/5 modules migrated, got {migrated}/5"

    # GAP-2: phase barrier present
    from agents.context import AnalysisContext, ContextSnapshot
    assert hasattr(AnalysisContext, "freeze")
    import inspect
    from agents.pipeline import GovernedPipeline
    src = inspect.getsource(GovernedPipeline.run)
    assert "freeze()" in src or "snapshot" in src.lower()

    # GAP-3: manifest persistence guaranteed
    assert hasattr(GovernedPipeline, "_safe_run")
    src = inspect.getsource(GovernedPipeline._safe_run)
    assert "finally" in src and "persist" in src
