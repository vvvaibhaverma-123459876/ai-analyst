"""
tests/test_brief_markdown_rendering.py

Reproduces and locks in the fix for bug 6: the executive brief rendered the
literal text "## What happened" instead of a heading.

Root cause: render_results() interpolated context.final_brief (markdown —
see InsightAgent._rule_brief / _llm_brief, both emit "## " headings and "- "
bullets) directly into an f-string HTML <div>...</div> and rendered the
whole thing in one st.markdown(..., unsafe_allow_html=True) call. Markdown
syntax is not reprocessed inside a raw HTML block, so the heading syntax
showed up as plain text. Fixed by emitting the wrapper tags and the brief
content as separate st.markdown() calls — the brief's own call has no
unsafe_allow_html, so Streamlit's normal markdown renderer processes it.

Needs streamlit — skipped under the lean requirements-ci.txt env, covered
by the demo-smoke CI job which installs requirements-demo.txt.
"""
from unittest.mock import patch

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from agents.context import AnalysisContext, AgentResult  # noqa: E402
import app.ui.product_app as product_app  # noqa: E402


def _render_and_capture_markdown_calls(context: AnalysisContext) -> list[tuple[tuple, dict]]:
    calls: list[tuple[tuple, dict]] = []

    def _record(*args, **kwargs):
        calls.append((args, kwargs))

    def _columns(spec, *a, **kw):
        n = spec if isinstance(spec, int) else len(spec)
        return [_DummyCol() for _ in range(n)]

    def _tabs(labels, *a, **kw):
        return [_DummyTab() for _ in range(len(labels))]

    with patch.object(product_app.st, "markdown", side_effect=_record), \
         patch.object(product_app.st, "columns", side_effect=_columns), \
         patch.object(product_app.st, "tabs", side_effect=_tabs), \
         patch.object(product_app.st, "metric"), \
         patch.object(product_app.st, "info"), \
         patch.object(product_app.st, "dataframe"), \
         patch.object(product_app.st, "plotly_chart"), \
         patch.object(product_app.st, "expander", return_value=_DummyTab()):
        product_app.render_results(context)
    return calls


class _DummyCol:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def metric(self, *a, **kw):
        pass


class _DummyTab:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _make_context(brief: str) -> AnalysisContext:
    ctx = AnalysisContext(
        df=pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "amount": range(30)}),
        date_col="date", kpi_col="amount",
    )
    ctx.final_brief = brief
    ctx.active_agents = ["trend"]
    return ctx


def test_brief_is_rendered_via_its_own_dedicated_markdown_call_without_html_wrapper():
    brief = "## What happened\n- trend: kpi up 12%\n\n## Why it happened\n- driver: channel mix"
    ctx = _make_context(brief)
    calls = _render_and_capture_markdown_calls(ctx)

    # Find the call whose content IS the brief text.
    brief_calls = [(args, kwargs) for args, kwargs in calls if args and args[0] == brief]
    assert brief_calls, f"no st.markdown call rendered the brief text verbatim; calls were: {calls}"

    # It must not be nested inside a raw HTML block: that call must not
    # pass unsafe_allow_html=True, and no *other* call may contain the
    # brief text embedded inside an HTML tag (the exact old bug shape).
    for args, kwargs in brief_calls:
        assert not kwargs.get("unsafe_allow_html"), (
            "brief content passed through unsafe_allow_html=True — markdown syntax "
            "inside it will render as literal text, not headings/bullets"
        )
    for args, kwargs in calls:
        if args and isinstance(args[0], str) and "<div" in args[0] and brief in args[0]:
            pytest.fail(f"brief text is embedded inside a raw HTML div: {args[0]!r}")


def test_real_rule_brief_output_renders_correctly():
    """End-to-end with the actual offline brief generator, not synthetic text."""
    from agents.insight_agent import InsightAgent

    ctx = _make_context("")
    ctx.results = {
        "trend": AgentResult(
            agent="trend", status="success", summary="KPI trending up 12% week over week.", data={},
        ),
    }
    real_brief = InsightAgent()._rule_brief(ctx, ctx.get_summaries())
    assert real_brief.startswith("## What happened"), "sanity check: the generator really does emit markdown headings"

    ctx.final_brief = real_brief
    calls = _render_and_capture_markdown_calls(ctx)
    brief_calls = [(args, kwargs) for args, kwargs in calls if args and args[0] == real_brief]
    assert brief_calls
    assert not brief_calls[0][1].get("unsafe_allow_html")
