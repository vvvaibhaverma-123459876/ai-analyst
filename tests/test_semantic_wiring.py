from agents.orchestrator_agent import OrchestratorAgent
from agents.context import AnalysisContext
from agents.trend_agent import TrendAgent
from semantic.join_graph import JoinGraph
import pandas as pd


def test_orchestrator_removes_root_cause_for_invalid_metric_dimension_combo():
    agent = OrchestratorAgent()
    ctx = AnalysisContext(
        business_context={"metric": "fd_payment_conversion"},
        data_profile={
            "rows": 100,
            "has_time_series": True,
            "has_funnel_signal": False,
            "has_cohort_signal": False,
            "dimensions": ["product"],
            "kpis": ["fd_payment_success"],
        },
    )
    result = agent.run(ctx)
    assert result.status == "success"
    assert "root_cause" not in result.data["plan"]


def test_join_graph_supports_list_style_join_config():
    graph = JoinGraph({
        "joins": [
            {
                "left_table": "events",
                "right_table": "daily_metrics",
                "join_type": "LEFT",
                "on": [{"left": "event_date", "right": "metric_date"}],
            }
        ]
    })
    path = graph.find_path("events", "daily_metrics")
    assert len(path) == 1
    assert path[0]["to"] == "daily_metrics"


def test_trend_agent_uses_grain_resolver_for_governed_metric():
    df = pd.DataFrame(
        {
            "event_date": pd.date_range("2026-01-01", periods=21, freq="D"),
            "converted_users": range(21),
        }
    )
    ctx = AnalysisContext(
        df=df,
        date_col="event_date",
        kpi_col="converted_users",
        grain="hourly",
        business_context={"metric": "conversion_rate"},
    )
    result = TrendAgent().run(ctx)
    assert result.status == "success"
    assert result.data["resolved_grain"] == "Daily"
