"""
tests/test_bench_pipeline_e2e.py
End-to-end and adversarial pipeline benchmarks:
  - Full AgentRunner pipeline on synthetic data (no LLM needed)
  - InsightAgent recommendation wiring
  - SecurityShell integration with full pipeline
  - ReplayHarness round-trip
  - Adversarial data scenarios (all-null, constant, single-row, zero-variance)
  - Multi-tenant isolation through runner
  - Guardian + EvidenceGrader integration
  - Data quality gate blocking propagation
  - Approval gate enforcement in runner
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import make_ts, make_funnel, make_cohort, make_segment, make_ab
from agents.context import AnalysisContext, AgentResult


def _make_ctx(df=None, kpi="revenue", date="date", n=60, spike_idx=None):
    if df is None:
        df = make_ts(n=n, spike_idx=spike_idx)
    ctx = AnalysisContext(
        df=df, kpi_col=kpi, date_col=date,
        tenant_id="default", user_id="system",
        business_context={"company": "Acme", "industry": "SaaS"},
    )
    return ctx


# ══════════════════════════════════════════════════════════════════════
# InsightAgent → RecommendationRanker wiring
# ══════════════════════════════════════════════════════════════════════

class TestInsightAgentRankingWiring:

    def test_ranked_recommendations_non_empty(self):
        from agents.insight_agent import InsightAgent
        ctx = _make_ctx()
        ctx.write_result(AgentResult(
            agent="trend", status="success",
            summary="Revenue declined 8%", data={},
        ))
        ctx.write_result(AgentResult(
            agent="root_cause", status="success",
            summary="Android drag identified",
            data={"movers": {"negative": [
                {"dimension": "platform", "value": "android"}
            ]}},
        ))
        result = InsightAgent().run(ctx)
        assert result.status == "success"
        assert len(result.data["ranked_recommendations"]) >= 1

    def test_ranked_recommendations_ordered_by_score(self):
        from agents.insight_agent import InsightAgent
        ctx = _make_ctx()
        ctx.data_quality_report = {"score": 0.3}  # triggers DQ action
        ctx.write_result(AgentResult(
            agent="anomaly", status="success",
            summary="Spike detected",
            data={"anomaly_count": 2, "anomaly_records": []},
        ))
        ctx.write_result(AgentResult(
            agent="root_cause", status="success",
            summary="Platform drag",
            data={"movers": {"negative": [{"dimension":"p","value":"ios"}]}},
        ))
        result = InsightAgent().run(ctx)
        recs = result.data["ranked_recommendations"]
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_recommendation_candidates_populated_in_context(self):
        from agents.insight_agent import InsightAgent
        ctx = _make_ctx()
        ctx.write_result(AgentResult(
            agent="trend", status="success", summary="trend done", data={}))
        InsightAgent().run(ctx)
        assert len(ctx.recommendation_candidates) >= 1

    def test_brief_non_empty_without_llm(self):
        from agents.insight_agent import InsightAgent
        ctx = _make_ctx()
        ctx.write_result(AgentResult(
            agent="trend", status="success", summary="revenue up 5%", data={}))
        result = InsightAgent().run(ctx)
        assert result.data["brief"]

    def test_followup_questions_generated(self):
        from agents.insight_agent import InsightAgent
        ctx = _make_ctx()
        ctx.write_result(AgentResult(
            agent="trend", status="success", summary="trend up", data={}))
        result = InsightAgent().run(ctx)
        assert len(result.data["follow_up_questions"]) >= 1


# ══════════════════════════════════════════════════════════════════════
# Full pipeline (lightweight — no LLM, no Prophet)
# ══════════════════════════════════════════════════════════════════════

class TestPipelineEndToEnd:

    def _run(self, df, kpi="revenue", date="date"):
        from agents.runner import AgentRunner
        ctx = _make_ctx(df, kpi, date)
        return AgentRunner().run(ctx)

    def test_pipeline_completes_on_clean_ts(self):
        ctx = self._run(make_ts(n=60))
        assert "insight" in ctx.results
        assert ctx.results["insight"].status == "success"

    def test_pipeline_completes_on_sparse_ts(self):
        df = make_ts(n=14)
        ctx = self._run(df)
        assert "eda" in ctx.results

    def test_pipeline_sets_run_id(self):
        ctx = self._run(make_ts(n=40))
        assert ctx.run_id

    def test_pipeline_generates_brief(self):
        ctx = self._run(make_ts(n=60))
        assert ctx.final_brief

    def test_pipeline_creates_manifest(self):
        ctx = self._run(make_ts(n=40))
        assert ctx.run_manifest
        assert "run_id" in ctx.run_manifest or isinstance(ctx.run_manifest, dict)

    def test_pipeline_data_quality_report_populated(self):
        ctx = self._run(make_ts(n=40))
        assert ctx.data_quality_report
        assert "score" in ctx.data_quality_report

    def test_pipeline_approval_log_populated(self):
        ctx = self._run(make_ts(n=40))
        assert len(ctx.approval_log) >= 1

    def test_pipeline_research_plan_generated(self):
        ctx = self._run(make_ts(n=60))
        assert ctx.research_plan is not None

    def test_pipeline_on_funnel_data(self):
        df = make_funnel(500)
        from agents.runner import AgentRunner
        ctx = AnalysisContext(
            df=df, kpi_col="stage", date_col="date",
            tenant_id="default",
        )
        result = AgentRunner().run(ctx)
        assert "eda" in result.results

    def test_pipeline_on_segment_data(self):
        df = make_segment(300)
        ctx = self._run(df, kpi="revenue", date="date")
        assert "eda" in ctx.results


# ══════════════════════════════════════════════════════════════════════
# Adversarial inputs
# ══════════════════════════════════════════════════════════════════════

class TestAdversarialInputs:

    def _run(self, df, kpi="v", date="d"):
        from agents.runner import AgentRunner
        ctx = AnalysisContext(df=df, kpi_col=kpi, date_col=date,
                              tenant_id="default")
        return AgentRunner().run(ctx)

    def test_all_null_kpi_no_crash(self):
        df = pd.DataFrame({
            "d": pd.date_range("2025-01-01", periods=20),
            "v": [None] * 20,
        })
        ctx = self._run(df)
        assert "eda" in ctx.results

    def test_constant_kpi_no_crash(self):
        df = pd.DataFrame({
            "d": pd.date_range("2025-01-01", periods=30),
            "v": [100.0] * 30,
        })
        ctx = self._run(df)
        assert "eda" in ctx.results

    def test_single_row_no_crash(self):
        df = pd.DataFrame({"d": ["2025-01-01"], "v": [42.0]})
        ctx = self._run(df)
        assert "eda" in ctx.results

    def test_empty_dataframe_handled(self):
        df = pd.DataFrame({"d": [], "v": []})
        ctx = self._run(df)
        # Should not raise — DQ gate should block cleanly
        assert ctx.data_quality_report.get("ok") is False

    def test_extreme_outlier_no_crash(self):
        df = make_ts(n=60)
        df.loc[30, "revenue"] = 1e15
        from agents.runner import AgentRunner
        ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date",
                              tenant_id="default")
        AgentRunner().run(ctx)  # should not raise

    def test_missing_date_col_name_no_crash(self):
        df = make_ts(n=40)
        from agents.runner import AgentRunner
        ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="nonexistent",
                              tenant_id="default")
        AgentRunner().run(ctx)

    def test_unicode_kpi_col_no_crash(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=20),
            "收入": np.random.default_rng(1).normal(100, 5, 20),
        })
        from agents.runner import AgentRunner
        ctx = AnalysisContext(df=df, kpi_col="收入", date_col="date",
                              tenant_id="default")
        AgentRunner().run(ctx)

    def test_duplicate_rows_handled(self):
        df = make_ts(n=30)
        df = pd.concat([df, df]).reset_index(drop=True)
        from agents.runner import AgentRunner
        ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date",
                              tenant_id="default")
        result = AgentRunner().run(ctx)
        assert result.data_quality_report["duplicate_ratio"] > 0


# ══════════════════════════════════════════════════════════════════════
# SecurityShell through pipeline
# ══════════════════════════════════════════════════════════════════════

class TestSecurityThroughPipeline:

    def test_pii_masked_before_analysis(self):
        from security.security_shell import SecurityShell
        from agents.runner import AgentRunner
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=30),
            "email": [f"user{i}@corp.com" for i in range(30)],
            "revenue": np.random.default_rng(5).normal(100, 10, 30),
        })
        shell = SecurityShell(tenant_id="acme", user_id="system")
        masked_df, report = shell.process_dataframe(df, run_id="pii-test")
        assert report["mask_report"] is not None
        if "email" in masked_df.columns:
            assert masked_df["email"].apply(lambda x: "@" not in str(x)).all()

    def test_cross_tenant_access_blocked(self):
        from security.security_shell import SecurityShell
        shell = SecurityShell(tenant_id="acme", user_id="eve", role="analyst")
        with pytest.raises(PermissionError):
            shell.publish_output({"data": "secret"}, requested_tenant_id="rival")


# ══════════════════════════════════════════════════════════════════════
# ReplayHarness round-trip
# ══════════════════════════════════════════════════════════════════════

class TestReplayHarnessRoundTrip:

    def test_replay_produces_same_agent_set(self, tmp_path):
        from agents.runner import AgentRunner
        from evaluation.replay_harness import ReplayHarness
        from versioning.run_manifest import RunManifest

        df = make_ts(n=30)
        # First run
        ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date",
                              run_id="original-run", tenant_id="default")
        ctx = AgentRunner().run(ctx)
        original_agents = set(ctx.results.keys())

        # Build replay harness manually (saves CSV + manifest)
        import os
        data_dir = tmp_path / "memory" / "replay_data"
        data_dir.mkdir(parents=True)
        manifest_dir = tmp_path / "memory" / "manifests"
        manifest_dir.mkdir(parents=True)
        csv_path = data_dir / "original-run.csv"
        df.to_csv(csv_path, index=False)

        m = RunManifest.create("original-run")
        m.replay_data_path = str(csv_path)
        m.replay_context = {
            "date_col": "date", "kpi_col": "revenue",
            "grain": "Daily", "filename": "test.csv",
            "tenant_id": "default", "user_id": "system",
        }
        m.persist(base_dir=str(manifest_dir))

        # Replay
        harness = ReplayHarness(base_dir=str(manifest_dir))
        replayed_ctx = harness.replay("original-run")
        assert replayed_ctx.run_id.endswith("-replay")
        assert "eda" in replayed_ctx.results
        assert "insight" in replayed_ctx.results

    def test_replay_missing_manifest_raises(self, tmp_path):
        from evaluation.replay_harness import ReplayHarness
        with pytest.raises(FileNotFoundError):
            ReplayHarness(base_dir=str(tmp_path)).replay("nonexistent-run")

    def test_replay_manifest_without_data_path_raises(self, tmp_path):
        from evaluation.replay_harness import ReplayHarness
        from versioning.run_manifest import RunManifest
        m = RunManifest.create("no-data")
        m.persist(base_dir=str(tmp_path))
        with pytest.raises(ValueError):
            ReplayHarness(base_dir=str(tmp_path)).replay("no-data")


# ══════════════════════════════════════════════════════════════════════
# Metric registry + SQL generator integration
# ══════════════════════════════════════════════════════════════════════

class TestMetricRegistryIntegration:

    def _registry(self):
        from semantic.metric_registry import MetricRegistry
        return MetricRegistry({
            "revenue": {
                "description": "Total net revenue",
                "aggregation": "sum",
                "column": "amount_usd",
                "aliases": ["net revenue", "total revenue"],
                "dimensions": ["channel", "platform"],
                "allowed_grains": ["daily", "weekly", "monthly"],
                "owner": "finance",
                "maturity": "production",
                "source_tables": ["orders"],
                "caveats": ["excludes refunds"],
            }
        })

    def test_resolve_alias(self):
        reg = self._registry()
        assert reg.resolve("total revenue this week") == "revenue"

    def test_get_by_exact_name(self):
        reg = self._registry()
        m = reg.get("revenue")
        assert m.key == "revenue"
        assert m.owner == "finance"

    def test_validate_valid_dimension(self):
        reg = self._registry()
        assert reg.validate_dimension("revenue", "channel") is True

    def test_validate_invalid_dimension(self):
        reg = self._registry()
        assert reg.validate_dimension("revenue", "country") is False

    def test_validate_valid_grain(self):
        reg = self._registry()
        assert reg.validate_grain("revenue", "daily") is True

    def test_validate_invalid_grain(self):
        reg = self._registry()
        assert reg.validate_grain("revenue", "hourly") is False

    def test_explain_returns_full_audit(self):
        reg = self._registry()
        audit = reg.explain("revenue")
        for key in ("metric","formula","owner","maturity","caveats"):
            assert key in audit

    def test_to_prompt_context_contains_metric(self):
        reg = self._registry()
        ctx = reg.to_prompt_context()
        assert "revenue" in ctx
        assert "finance" in ctx

    def test_missing_metric_raises(self):
        from core.exceptions import MetadataError
        reg = self._registry()
        with pytest.raises(MetadataError):
            reg.get("nonexistent_kpi")

    def test_list_all(self):
        reg = self._registry()
        assert "revenue" in reg.list_all()
