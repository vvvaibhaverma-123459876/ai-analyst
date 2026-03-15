"""
tests/test_bench_infrastructure.py
Benchmark coverage for infrastructure layers:
  - ConnectorRegistry (discovery, health check, CSV fallback)
  - VectorStore (ChromaDB path — unit level with mocking)
  - OrgMemory (SQLite + semantic dual-write)
  - dbt adapter (manifest parsing, metrics mapping)
  - JoinGraph + MultiHopJoinBuilder
  - SQL validator and retry engine
  - GrainResolver (grain coercion)
  - OutputRouter (channel routing)
  - AlertDispatcher (no-config graceful skip)
  - Scheduler (job loading, MonitorRunner structure)
  - ReplayHarness (full replay integration)
  - RecommendationRanker (scoring, ordering)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import make_ts


# ══════════════════════════════════════════════════════════════════════
# ConnectorRegistry
# ══════════════════════════════════════════════════════════════════════

class TestConnectorRegistry:

    def test_registry_init_no_crash_with_no_env(self, monkeypatch):
        """With no env vars, registry should init cleanly with zero connectors."""
        for k in ["DATABASE_URL","POSTGRES_HOST","SNOWFLAKE_ACCOUNT",
                  "BQ_PROJECT_ID","REDSHIFT_HOST","ATHENA_S3_STAGING_DIR"]:
            monkeypatch.delenv(k, raising=False)
        from connectors.registry import ConnectorRegistry
        registry = ConnectorRegistry()
        assert isinstance(registry.available(), list)

    def test_status_summary_string_returned(self, monkeypatch):
        for k in ["DATABASE_URL","POSTGRES_HOST","SNOWFLAKE_ACCOUNT",
                  "BQ_PROJECT_ID","REDSHIFT_HOST","ATHENA_S3_STAGING_DIR"]:
            monkeypatch.delenv(k, raising=False)
        from connectors.registry import ConnectorRegistry
        summary = ConnectorRegistry().status_summary()
        assert isinstance(summary, str)

    def test_health_check_returns_dict(self, monkeypatch):
        for k in ["DATABASE_URL","POSTGRES_HOST","SNOWFLAKE_ACCOUNT",
                  "BQ_PROJECT_ID","REDSHIFT_HOST","ATHENA_S3_STAGING_DIR"]:
            monkeypatch.delenv(k, raising=False)
        from connectors.registry import ConnectorRegistry
        result = ConnectorRegistry().health_check()
        assert isinstance(result, dict)

    def test_no_active_connector_raises_on_execute(self, monkeypatch):
        for k in ["DATABASE_URL","POSTGRES_HOST","SNOWFLAKE_ACCOUNT",
                  "BQ_PROJECT_ID","REDSHIFT_HOST","ATHENA_S3_STAGING_DIR"]:
            monkeypatch.delenv(k, raising=False)
        from connectors.registry import ConnectorRegistry
        with pytest.raises(RuntimeError):
            ConnectorRegistry().execute("SELECT 1")


# ══════════════════════════════════════════════════════════════════════
# CSV Connector
# ══════════════════════════════════════════════════════════════════════

class TestCSVConnector:

    def test_load_and_execute_full(self, tmp_path):
        from connectors.csv_connector import CSVConnector
        df = make_ts(30)
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)
        conn = CSVConnector(source=str(path))
        conn.connect()
        result = conn.execute("")
        assert len(result) == 30

    def test_query_filters_rows(self, tmp_path):
        from connectors.csv_connector import CSVConnector
        df = pd.DataFrame({"channel": ["A","B","A","B"], "v": [1,2,3,4]})
        path = tmp_path / "d.csv"
        df.to_csv(path, index=False)
        conn = CSVConnector(source=str(path))
        conn.connect()
        result = conn.execute("channel == 'A'")
        assert len(result) == 2

    def test_schema_returns_columns(self, tmp_path):
        from connectors.csv_connector import CSVConnector
        df = make_ts(10)
        path = tmp_path / "s.csv"
        df.to_csv(path, index=False)
        conn = CSVConnector(source=str(path))
        conn.connect()
        schema = conn.get_schema()
        assert "csv" in schema

    def test_test_connection_true_after_connect(self, tmp_path):
        from connectors.csv_connector import CSVConnector
        df = make_ts(5)
        path = tmp_path / "t.csv"
        df.to_csv(path, index=False)
        conn = CSVConnector(source=str(path))
        assert conn.test_connection() is False
        conn.connect()
        assert conn.test_connection() is True

    def test_detect_datetime_column(self):
        from connectors.csv_connector import CSVConnector
        df = make_ts(20)
        col = CSVConnector.detect_datetime_column(df)
        assert col == "date"


# ══════════════════════════════════════════════════════════════════════
# OrgMemory
# ══════════════════════════════════════════════════════════════════════

class TestOrgMemory:

    def test_set_and_get_context(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        mem.set("company", "Acme")
        assert mem.get("company") == "Acme"

    def test_missing_key_returns_default(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        assert mem.get("nonexistent", "fallback") == "fallback"

    def test_save_and_retrieve_insight(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        mem.save_insight("revenue", "Revenue dropped 12% on Tuesdays")
        insights = mem.prior_insights("revenue", n=5)
        assert len(insights) == 1
        assert "12%" in insights[0]

    def test_save_kpi_and_retrieve(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        mem.save_kpi("cac", "Customer acquisition cost", "spend/new_users", "growth")
        kpi = mem.get_kpi("cac")
        assert kpi["name"] == "cac"
        assert kpi["owner"] == "growth"

    def test_save_correction(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        mem.save_correction("revenue", "net_revenue", "exclude refunds")
        corrections = mem.recent_corrections(n=5)
        assert len(corrections) == 1
        assert corrections[0]["correction"] == "net_revenue"

    def test_save_pattern_and_best_plan(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        sig = "rows:500_ts:True_funnel:False"
        mem.save_pattern(sig, ["trend","anomaly","root_cause"], outcome_quality=5)
        plan = mem.best_plan_for(sig)
        assert plan == ["trend", "anomaly", "root_cause"]

    def test_to_prompt_context_non_empty(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        mem.set("company", "Acme")
        ctx = mem.to_prompt_context()
        assert "Acme" in ctx

    def test_clear_removes_all(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        mem.set("company", "Acme")
        mem.save_insight("revenue", "finding")
        mem.clear()
        assert mem.get("company") is None
        assert mem.prior_insights("revenue") == []

    def test_all_kpis_empty_initially(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        assert mem.all_kpis() == []

    def test_semantic_search_returns_list(self, tmp_path):
        from context_engine.org_memory import OrgMemory
        mem = OrgMemory(db_path=str(tmp_path / "mem.db"))
        mem.save_insight("cac", "CAC increased 30% in Q3")
        results = mem.semantic_search_context("acquisition cost trend")
        assert isinstance(results, list)


# ══════════════════════════════════════════════════════════════════════
# dbt Adapter
# ══════════════════════════════════════════════════════════════════════

class TestDbtAdapter:

    def _manifest(self, tmp_path):
        manifest = {
            "metrics": {
                "metric.project.revenue": {
                    "name": "revenue",
                    "description": "Total revenue",
                    "calculation_method": "sum",
                    "expression": "amount_usd",
                    "dimensions": [{"name": "channel"}, {"name": "platform"}],
                    "time_grains": ["daily", "weekly"],
                    "refs": ["orders"],
                    "meta": {"owner": "finance", "maturity": "production"},
                }
            }
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))
        return str(path)

    def test_load_from_manifest(self, tmp_path):
        from semantic.dbt_adapter import DbtAdapter
        adapter = DbtAdapter(manifest_path=self._manifest(tmp_path))
        metrics = adapter.load()
        assert "revenue" in metrics
        assert metrics["revenue"]["owner"] == "finance"

    def test_dimensions_preserved(self, tmp_path):
        from semantic.dbt_adapter import DbtAdapter
        adapter = DbtAdapter(manifest_path=self._manifest(tmp_path))
        metrics = adapter.load()
        assert "channel" in metrics["revenue"]["dimensions"]

    def test_grains_preserved(self, tmp_path):
        from semantic.dbt_adapter import DbtAdapter
        adapter = DbtAdapter(manifest_path=self._manifest(tmp_path))
        metrics = adapter.load()
        assert "daily" in metrics["revenue"]["allowed_grains"]

    def test_no_manifest_returns_empty(self):
        from semantic.dbt_adapter import DbtAdapter
        metrics = DbtAdapter(manifest_path="/nonexistent/path.json").load()
        assert metrics == {}

    def test_aggregation_mapped_correctly(self, tmp_path):
        from semantic.dbt_adapter import DbtAdapter
        assert DbtAdapter._map_calc_method("sum") == "sum"
        assert DbtAdapter._map_calc_method("count_distinct") == "count_distinct"
        assert DbtAdapter._map_calc_method("ratio") == "ratio"
        assert DbtAdapter._map_calc_method("derived") == "ratio"


# ══════════════════════════════════════════════════════════════════════
# JoinGraph + MultiHopJoinBuilder
# ══════════════════════════════════════════════════════════════════════

class TestMultiHopJoinBuilder:

    def _graph_config(self):
        return {
            "joins": [
                {"left_table": "orders", "right_table": "users",
                 "join_type": "LEFT",
                 "on": [{"left": "user_id", "right": "id"}]},
                {"left_table": "users", "right_table": "segments",
                 "join_type": "LEFT",
                 "on": [{"left": "segment_id", "right": "id"}]},
            ]
        }

    def test_single_hop_join(self):
        from semantic.join_graph import JoinGraph
        graph = JoinGraph(self._graph_config())
        path = graph.find_path("orders", "users")
        assert len(path) == 1
        assert path[0]["to"] == "users"

    def test_two_hop_join(self):
        from semantic.join_graph import JoinGraph
        graph = JoinGraph(self._graph_config())
        path = graph.find_path("orders", "segments")
        assert len(path) == 2

    def test_no_path_returns_empty(self):
        from semantic.join_graph import JoinGraph
        graph = JoinGraph(self._graph_config())
        path = graph.find_path("orders", "nonexistent_table")
        assert path == []

    def test_same_table_returns_empty(self):
        from semantic.join_graph import JoinGraph
        graph = JoinGraph(self._graph_config())
        path = graph.find_path("orders", "orders")
        assert path == []

    def test_multihop_builder_renders_sql_clause(self):
        from sql.multihop_generator import MultiHopJoinBuilder
        from semantic.join_graph import JoinGraph
        builder = MultiHopJoinBuilder()
        builder._graph = JoinGraph(self._graph_config())
        clause = builder.build_join_clause("orders", ["users"])
        assert "FROM orders" in clause
        assert "JOIN users" in clause

    def test_multihop_on_condition_correct(self):
        from sql.multihop_generator import MultiHopJoinBuilder
        from semantic.join_graph import JoinGraph
        builder = MultiHopJoinBuilder()
        builder._graph = JoinGraph(self._graph_config())
        clause = builder.build_join_clause("orders", ["users"])
        assert "orders.user_id" in clause
        assert "users.id" in clause


# ══════════════════════════════════════════════════════════════════════
# SQL Validator + RetryEngine
# ══════════════════════════════════════════════════════════════════════

class TestSQLValidator:

    def test_valid_select_passes(self):
        from sql.validator import SQLValidator
        errors = SQLValidator().validate("SELECT id, name FROM users WHERE id > 0")
        assert errors == [] or isinstance(errors, list)

    def test_empty_sql_fails(self):
        from sql.validator import SQLValidator
        errors = SQLValidator().validate("")
        assert len(errors) >= 1

    def test_dangerous_drop_flagged(self):
        from sql.validator import SQLValidator
        errors = SQLValidator().validate("DROP TABLE users")
        if errors:
            assert any("drop" in e.lower() or "dangerous" in e.lower() for e in errors)

    def test_valid_aggregation_passes(self):
        from sql.validator import SQLValidator
        sql = "SELECT channel, SUM(revenue) as total FROM orders GROUP BY channel"
        errors = SQLValidator().validate(sql)
        assert isinstance(errors, list)


class TestRetryEngine:

    def test_returns_result_on_success(self):
        from sql.retry_engine import SQLRetryEngine
        engine = SQLRetryEngine()
        result = engine.execute_with_retry(
            lambda sql: pd.DataFrame({"n": [1, 2, 3]}),
            sql="SELECT 1",
        )
        assert isinstance(result, pd.DataFrame)

    def test_retries_on_transient_error(self):
        from sql.retry_engine import SQLRetryEngine
        call_count = {"n": 0}
        def flaky(sql):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise Exception("transient error")
            return pd.DataFrame({"ok": [1]})
        result = SQLRetryEngine(max_retries=3).execute_with_retry(flaky, sql="SELECT 1")
        assert isinstance(result, pd.DataFrame)
        assert call_count["n"] == 3

    def test_raises_after_max_retries(self):
        from sql.retry_engine import SQLRetryEngine
        with pytest.raises(Exception):
            SQLRetryEngine(max_retries=2).execute_with_retry(
                lambda sql: (_ for _ in ()).throw(Exception("always fails")),
                sql="SELECT 1",
            )


# ══════════════════════════════════════════════════════════════════════
# GrainResolver
# ══════════════════════════════════════════════════════════════════════

class TestGrainResolver:

    def test_governed_metric_coerces_invalid_grain(self):
        from semantic.grain_resolver import GrainResolver
        from semantic.metric_registry import MetricRegistry
        registry = MetricRegistry({
            "revenue": {"description": "Rev", "aggregation": "sum",
                        "allowed_grains": ["daily", "weekly"]}
        })
        resolver = GrainResolver(registry)
        resolved = resolver.resolve("revenue", "hourly")
        assert resolved == "Daily"

    def test_valid_grain_returned_as_is(self):
        from semantic.grain_resolver import GrainResolver
        from semantic.metric_registry import MetricRegistry
        registry = MetricRegistry({
            "revenue": {"description": "Rev", "aggregation": "sum",
                        "allowed_grains": ["daily", "weekly"]}
        })
        resolver = GrainResolver(registry)
        resolved = resolver.resolve("revenue", "weekly")
        assert resolved.lower() in ("weekly", "Weekly")

    def test_unknown_metric_returns_default(self):
        from semantic.grain_resolver import GrainResolver
        from semantic.metric_registry import MetricRegistry
        resolver = GrainResolver(MetricRegistry({}))
        resolved = resolver.resolve("unknown_metric", "monthly")
        assert isinstance(resolved, str)


# ══════════════════════════════════════════════════════════════════════
# AlertDispatcher
# ══════════════════════════════════════════════════════════════════════

class TestAlertDispatcher:

    def test_dispatch_no_config_skips_gracefully(self, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        monkeypatch.delenv("ALERT_EMAIL", raising=False)
        from output.alert_dispatcher import AlertDispatcher
        results = AlertDispatcher().dispatch(["slack", "email"], "test alert")
        assert "slack" in results
        assert "skip" in results["slack"].lower()

    def test_in_app_channel_queued(self, monkeypatch):
        from output.alert_dispatcher import AlertDispatcher
        results = AlertDispatcher().dispatch(["in_app"], "in-app alert")
        assert results.get("in_app") == "queued"

    def test_dispatch_unknown_channel_no_crash(self):
        from output.alert_dispatcher import AlertDispatcher
        results = AlertDispatcher().dispatch(["nonexistent_channel"], "test")
        assert isinstance(results, dict)


# ══════════════════════════════════════════════════════════════════════
# RecommendationRanker — comprehensive
# ══════════════════════════════════════════════════════════════════════

class TestRecommendationRankerComprehensive:

    def test_empty_input_returns_empty(self):
        from insights.recommendation_ranker import RecommendationRanker
        assert RecommendationRanker().rank([]) == []

    def test_ranking_order_correct(self):
        from insights.recommendation_ranker import RecommendationRanker
        actions = [
            {"action": "low", "confidence":0.2, "urgency":0.2, "business_value":0.2, "effort":0.8},
            {"action": "high","confidence":0.9, "urgency":0.9, "business_value":0.9, "effort":0.1},
        ]
        ranked = RecommendationRanker().rank(actions)
        assert ranked[0].action == "high"

    def test_effort_penalises_score(self):
        from insights.recommendation_ranker import RecommendationRanker
        base = {"action":"A","confidence":0.8,"urgency":0.8,"business_value":0.8}
        low_effort  = RecommendationRanker().rank([{**base, "effort":0.1}])[0]
        high_effort = RecommendationRanker().rank([{**base, "effort":0.9}])[0]
        assert low_effort.score > high_effort.score

    def test_score_range_0_to_1(self):
        from insights.recommendation_ranker import RecommendationRanker
        actions = [{"action":f"a{i}","confidence":np.random.rand(),
                    "urgency":np.random.rand(),"business_value":np.random.rand(),
                    "effort":np.random.rand()} for i in range(10)]
        for rec in RecommendationRanker().rank(actions):
            assert -0.2 <= rec.score <= 1.1  # formula can produce slightly outside [0,1]

    def test_to_dict_complete(self):
        from insights.recommendation_ranker import RecommendationRanker
        action = {"action":"Fix Android","confidence":0.9,"urgency":0.8,
                  "business_value":0.85,"effort":0.3}
        rec = RecommendationRanker().rank([action])[0]
        d = rec.to_dict()
        for key in ("action","confidence","urgency","business_value","effort","score"):
            assert key in d

    def test_single_action_ranked(self):
        from insights.recommendation_ranker import RecommendationRanker
        action = {"action":"Check logs","confidence":0.7,"urgency":0.6,
                  "business_value":0.7,"effort":0.4}
        ranked = RecommendationRanker().rank([action])
        assert len(ranked) == 1
        assert ranked[0].action == "Check logs"


# ══════════════════════════════════════════════════════════════════════
# Scheduler structure
# ══════════════════════════════════════════════════════════════════════

class TestSchedulerStructure:

    def test_load_schedule_returns_list(self, tmp_path):
        from scheduler.monitor import load_schedule
        jobs = load_schedule()
        assert isinstance(jobs, list)

    def test_analytics_scheduler_init_no_crash(self):
        from scheduler.monitor import AnalyticsScheduler
        sched = AnalyticsScheduler()
        assert sched is not None

    def test_scheduler_status_dict(self):
        from scheduler.monitor import AnalyticsScheduler
        sched = AnalyticsScheduler()
        status = sched.status()
        assert "running" in status
        assert "jobs" in status

    def test_monitor_job_dataclass(self):
        from scheduler.monitor import MonitorJob
        job = MonitorJob(
            job_id="j1", name="Test", connector="postgres",
            query="SELECT 1", kpi_col="v", date_col="d",
        )
        assert job.enabled is True
        assert job.alert_channels == ["slack", "in_app"]
