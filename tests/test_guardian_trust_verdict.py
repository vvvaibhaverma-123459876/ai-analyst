"""
tests/test_guardian_trust_verdict.py

Reproduces and locks in the fix for bug 5: the observed run showed eda
errored, nlp/vision errored, forecast skipped, anomaly-jury confidence 0.10,
and the self-contradictory headline "KPI increased by +2,530 (+0.0%)" (DoD/
WoW/MoM all +0.0%) — yet the "Trust review" tile read "success", because it
displayed guardian.status (did the guardian program itself crash — always
"success") rather than any actual judgment about whether the run could be
trusted.

Two independent fixes, tested separately:
  1. analysis/statistics.py + analysis/root_cause.py + agents/trend_agent.py:
     pct-change is now honest about a zero/missing baseline ("n/a (new
     baseline)") instead of fabricating "+0.0%" for a real, nonzero delta.
  2. guardian/guardian_agent.py: GuardianAgent._assess_run_health() computes
     a real run_verdict (success/degraded/failed) from agent errors, KPI
     delta/pct-change inconsistency, a collapsed time series, and low jury
     confidence — surfaced via guardian.data["run_verdict"] /
     ["downgrade_reasons"], which the Command Center tile and the executive
     brief's Confidence assessment section now read instead of the
     always-"success" agent status.
"""
import pandas as pd
import pytest

from agents.context import AnalysisContext, AgentResult
from analysis.statistics import safe_pct_change, format_pct_change
from guardian.guardian_agent import GuardianAgent


# ── Fix 1: honest pct-change ─────────────────────────────────────────

def test_safe_pct_change_zero_baseline_nonzero_delta_is_none():
    assert safe_pct_change(delta=2530.0, baseline=0.0) is None


def test_safe_pct_change_both_zero_is_zero():
    assert safe_pct_change(delta=0.0, baseline=0.0) == 0.0


def test_safe_pct_change_normal_case():
    assert safe_pct_change(delta=50.0, baseline=200.0) == 25.0


def test_format_pct_change_honest_about_new_baseline():
    assert format_pct_change(None) == "n/a (new baseline)"
    assert format_pct_change(0.0) == "+0.0%"
    assert format_pct_change(-12.3) == "-12.3%"


def test_root_cause_reproduces_and_fixes_the_exact_reported_headline():
    """The exact reported scenario: 2,530 successes concentrated with no
    activity in the prior comparison window."""
    from analysis.root_cause import RootCauseAnalyzer

    df = pd.DataFrame({
        "event_time": ["2026-02-01"] * 18965,
        "payment_success_event": [1] * 2530 + [0] * 16435,
    })
    result = RootCauseAnalyzer().analyze(df, date_col="event_time", kpi_col="payment_success_event", days=3)
    assert "+0.0%" not in result.summary, f"summary must not fabricate a 0% change: {result.summary!r}"
    assert "n/a" in result.summary or "new baseline" in result.summary
    assert result.metadata["pct_change_raw"] is None
    assert result.metadata["delta"] == 2530.0


# ── Fix 2: guardian run_verdict ──────────────────────────────────────

def _base_context(**overrides) -> AnalysisContext:
    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "amount": range(30)})
    defaults = dict(df=df, date_col="date", kpi_col="amount", tenant_id="default", user_id="test")
    defaults.update(overrides)
    return AnalysisContext(**defaults)


def _success(agent: str, data: dict | None = None) -> AgentResult:
    return AgentResult(agent=agent, status="success", summary=f"{agent} ok", data=data or {})


def _error(agent: str) -> AgentResult:
    return AgentResult(agent=agent, status="error", summary=f"{agent} crashed", data={}, error="boom")


def test_clean_run_is_success_with_no_reasons():
    ctx = _base_context()
    ctx.results = {"trend": _success("trend"), "guardian": _success("guardian")}
    ctx.ts = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "amount": range(10)})
    status, reasons = GuardianAgent()._assess_run_health(ctx)
    assert status == "success"
    assert reasons == []


def test_any_agent_error_downgrades_to_failed():
    ctx = _base_context()
    ctx.results = {
        "eda": _error("eda"),
        "nlp": _error("nlp"),
        "vision": _error("vision"),
        "trend": _success("trend"),
    }
    ctx.ts = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "amount": range(10)})
    status, reasons = GuardianAgent()._assess_run_health(ctx)
    assert status == "failed"
    assert any("eda" in r and "nlp" in r and "vision" in r for r in reasons)


def test_inconsistent_delta_pct_downgrades_to_degraded():
    ctx = _base_context()
    ctx.results = {
        "root_cause": _success("root_cause", {"delta": 2530.0, "pct_change_raw": None, "pct_change": 0.0}),
    }
    ctx.ts = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "amount": range(10)})
    status, reasons = GuardianAgent()._assess_run_health(ctx)
    assert status == "degraded"
    assert any("new-baseline" in r or "no baseline" in r for r in reasons)


def test_collapsed_trend_downgrades_to_degraded():
    ctx = _base_context()
    ctx.results = {"trend": _success("trend")}
    ctx.ts = pd.DataFrame({"date": ["2026-02-01"], "amount": [100]})  # 1 point
    status, reasons = GuardianAgent()._assess_run_health(ctx)
    assert status == "degraded"
    assert any("collapsed" in r for r in reasons)


def test_low_jury_confidence_downgrades_to_degraded():
    ctx = _base_context()
    ctx.results = {
        "anomaly": _success("anomaly", {"confidence": 0.10}),
    }
    ctx.ts = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "amount": range(10)})
    status, reasons = GuardianAgent()._assess_run_health(ctx)
    assert status == "degraded"
    assert any("0.10" in r for r in reasons)


def test_the_exact_reported_scenario_is_no_longer_a_silent_success():
    """All four reported symptoms together: errored agents, a collapsed
    trend, low jury confidence, and the inconsistent headline number."""
    ctx = _base_context()
    ctx.results = {
        "eda": _error("eda"),
        "nlp": _error("nlp"),
        "vision": _error("vision"),
        "anomaly": _success("anomaly", {"confidence": 0.10}),
        "root_cause": _success("root_cause", {"delta": 2530.0, "pct_change_raw": None, "pct_change": 0.0}),
    }
    ctx.ts = pd.DataFrame({"date": ["2026-02-01"], "amount": [2530]})
    status, reasons = GuardianAgent()._assess_run_health(ctx)
    assert status == "failed"  # agent errors are the hardest signal
    assert len(reasons) >= 3


def test_full_guardian_run_exposes_run_verdict_in_agent_result():
    """End-to-end through GuardianAgent.run() (not just the private
    assessment helper), confirming the public data contract the UI and
    brief actually read from."""
    ctx = _base_context()
    ctx.results = {
        "eda": _error("eda"),
        "trend": _success("trend"),
    }
    ctx.ts = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=10), "amount": range(10)})
    ctx.active_agents = ["eda", "trend"]
    result = GuardianAgent().run(ctx)
    assert result.status == "success", "the guardian program itself must not crash just because the run it's reviewing is bad"
    assert result.data["run_verdict"] == "failed"
    assert result.data["downgrade_reasons"]
    assert result.data["approved"] is False


def test_degraded_verdict_is_surfaced_in_the_executive_brief():
    from agents.insight_agent import InsightAgent

    ctx = _base_context()
    ctx.results = {
        "trend": _success("trend", {}),
        "guardian": _success("guardian", {
            "run_verdict": "degraded",
            "downgrade_reasons": ["low jury confidence: anomaly=0.10."],
        }),
    }
    brief = InsightAgent()._rule_brief(ctx, ctx.get_summaries())
    assert "degraded" in brief
    assert "low jury confidence" in brief
