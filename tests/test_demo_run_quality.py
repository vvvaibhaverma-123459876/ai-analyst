"""
tests/test_demo_run_quality.py

Bug 7: the end-to-end regression gate. Runs the FULL governed pipeline, in
offline mode with a live socket-connect block (CI has no keys and must never
reach the network either way), against BOTH bundled demo datasets, and
asserts every symptom from the recorded failure mode is gone:

  - bug 2: the auto-selected date/KPI/funnel-stage/user-id columns are
    exactly correct for both datasets (not a numeric flag column, not a
    timestamp mistaken for a funnel stage).
  - bug 3: the trend time series has >= 10 points, not a single
    epoch-collapsed bucket.
  - bug 4: zero agent status == "error" (skips are fine, but only with a
    real, data-driven reason — never an unhandled crash).
  - bug 5: the guardian's run_verdict is "success" with no downgrade
    reasons on this known-clean bundled data, and the reported delta/
    pct-change are mutually consistent (no fabricated "+0.0%" for a
    nonzero delta).
  - bug 6: the executive brief renders as real markdown, not literal "##"
    text, when it goes through render_results().

This is the contract: the exact recorded failure mode (auto-picked
vkyc_success_event as the date column, funnel stage = event_time, a
1-bucket trend, three crashed agents, "Trust review: success" over a
self-contradictory headline, and literal "## What happened" on screen) can
never ship again without this test catching it first.

Needs streamlit — skipped under the lean requirements-ci.txt env, covered
by the demo-smoke CI job which installs requirements-demo.txt and runs the
full suite (not just this file) precisely so tests like this one actually
execute in CI instead of being silently skipped.
"""
import socket

import pytest

pytest.importorskip("streamlit")

from agents.context import AnalysisContext  # noqa: E402
from agents.pipeline import GovernedPipeline  # noqa: E402
import app.ui.product_app as product_app  # noqa: E402


EXPECTED_COLUMNS = {
    "Fintech onboarding demo": {
        "filename": "sample_fintech_onboarding.csv",
        "date_col": "event_time",
        "kpi_col": "payment_success_event",
        "user_col": "user_id",
        "stage_col": "stage",
        # 18,965 rows: enough statistical power that the anomaly jury
        # should be genuinely confident — this dataset must be fully clean.
        "require_clean_verdict": True,
    },
    "Growth KPI demo": {
        "filename": "sample_growth_data.csv",
        "date_col": "date",
        "kpi_col": "revenue",
        "user_col": None,
        "stage_col": None,
        # Only 30 rows / 10 daily points — genuinely too little data for
        # the anomaly jury to be statistically confident. A "degraded"
        # verdict from *that* is an honest signal, not a regression; this
        # dataset only needs to be free of the *specific* reported failure
        # modes (agent crashes, delta/pct inconsistency, a collapsed trend).
        "require_clean_verdict": False,
    },
}


@pytest.fixture
def block_network(monkeypatch):
    def _blocked(self, *a, **kw):
        raise AssertionError(f"offline demo run must not open network connections; blocked connect({a!r})")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)


@pytest.mark.parametrize("dataset_label", list(EXPECTED_COLUMNS.keys()))
def test_full_demo_run_meets_the_quality_contract(dataset_label, block_network, monkeypatch):
    monkeypatch.setenv("ANALYST_MODE", "offline")
    expected = EXPECTED_COLUMNS[dataset_label]

    # ── bug 2: column auto-inference ─────────────────────────────
    df = product_app.load_sample(dataset_label)
    guess = product_app.resolve_columns(df, expected["filename"])
    assert guess.date_col == expected["date_col"], f"{dataset_label}: wrong auto date column"
    assert guess.kpi_col == expected["kpi_col"], f"{dataset_label}: wrong auto KPI column"
    assert guess.user_col == expected["user_col"], f"{dataset_label}: wrong auto user-id column"
    assert guess.stage_col == expected["stage_col"], f"{dataset_label}: wrong auto funnel-stage column"

    ctx = AnalysisContext(
        df=df, date_col=guess.date_col, kpi_col=guess.kpi_col, grain="Daily",
        filename=expected["filename"],
        business_context={"tenant_id": "default", "user_id": "demo-run-quality", "primary_goal": "test"},
        tenant_id="default", user_id="demo-run-quality",
    )
    finished = GovernedPipeline().run(ctx)

    # ── bug 3: trend not collapsed ────────────────────────────────
    assert len(finished.ts) >= 10, (
        f"{dataset_label}: trend time series has only {len(finished.ts)} point(s) — "
        "looks like an epoch-collapsed or otherwise degenerate date axis"
    )

    # ── bug 4: zero agent crashes ─────────────────────────────────
    errored = {name: r.error for name, r in finished.results.items() if r.status == "error"}
    assert not errored, f"{dataset_label}: agent(s) crashed: {errored}"

    # ── bug 5: honest, clean guardian verdict ─────────────────────
    guardian = finished.results.get("guardian")
    assert guardian is not None and guardian.status == "success", f"{dataset_label}: guardian did not run"
    reasons = guardian.data["downgrade_reasons"]
    if expected["require_clean_verdict"]:
        assert guardian.data["run_verdict"] == "success", (
            f"{dataset_label}: expected a fully clean run_verdict on this large, known-good "
            f"dataset, got {guardian.data['run_verdict']!r}: {reasons}"
        )
        assert reasons == []
    else:
        # Never allow the *specific* reported failure modes back, even if
        # this smaller dataset can legitimately still be "degraded" for an
        # honest reason (e.g. low statistical power for anomaly detection).
        assert not any("errored" in r for r in reasons), f"{dataset_label}: {reasons}"
        assert not any("new-baseline" in r or "inconsistent" in r for r in reasons), f"{dataset_label}: {reasons}"
        assert not any("collapsed" in r for r in reasons), f"{dataset_label}: {reasons}"

    rc = finished.results.get("root_cause")
    if rc and rc.status == "success":
        delta = rc.data.get("delta", 0.0)
        pct_raw = rc.data.get("pct_change_raw")
        # The exact reported inconsistency: a nonzero delta with a literal
        # "0.0%" (rather than the honest "n/a (new baseline)") must never
        # happen again.
        assert not (delta != 0 and pct_raw == 0.0 and rc.data.get("pct_change") == 0.0 and "0.0%" in rc.summary and "n/a" not in rc.summary), (
            f"{dataset_label}: inconsistent headline — delta={delta} but summary claims 0% change: {rc.summary!r}"
        )

    # ── bug 6: brief renders as real markdown, not literal text ───
    assert finished.final_brief, f"{dataset_label}: no executive brief was generated"
    assert finished.final_brief.startswith("## "), "sanity check: the brief generator emits markdown"

    from unittest.mock import patch

    class _DummyCol:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def metric(self, *a, **kw): pass

    class _DummyTab:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    markdown_calls = []
    with patch.object(product_app.st, "markdown", side_effect=lambda *a, **kw: markdown_calls.append((a, kw))), \
         patch.object(product_app.st, "columns", side_effect=lambda spec, *a, **kw: [_DummyCol() for _ in range(spec if isinstance(spec, int) else len(spec))]), \
         patch.object(product_app.st, "tabs", side_effect=lambda labels, *a, **kw: [_DummyTab() for _ in labels]), \
         patch.object(product_app.st, "metric"), \
         patch.object(product_app.st, "info"), \
         patch.object(product_app.st, "dataframe"), \
         patch.object(product_app.st, "plotly_chart"), \
         patch.object(product_app.st, "expander", return_value=_DummyTab()):
        product_app.render_results(finished)

    brief_calls = [(a, kw) for a, kw in markdown_calls if a and a[0] == finished.final_brief]
    assert brief_calls, "the real final_brief was never rendered via its own st.markdown() call"
    assert not brief_calls[0][1].get("unsafe_allow_html"), (
        "the real final_brief was rendered with unsafe_allow_html=True — its markdown "
        "headings/bullets would show up as literal text, not real formatting"
    )
    for a, kw in markdown_calls:
        if a and isinstance(a[0], str) and "<div" in a[0] and "## " in a[0]:
            pytest.fail(f"brief markdown leaked into a raw HTML block: {a[0][:200]!r}")
