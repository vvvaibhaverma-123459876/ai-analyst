"""
tests/test_context_snapshot_contract.py

Reproduces and locks in the fix for bug 4's ContextSnapshot half: nlp and
vision failed with 'ContextSnapshot' object has no attribute 'document' —
AnalysisContext.freeze() (agents/context.py) builds a ContextSnapshot for
Phase 5's parallel agents but never copied the `document` field, and
ContextSnapshot didn't even declare it, even though nlp_agent.py and
vision_agent.py both read context.document.

This is a whole class of bug — any parallel agent reading a live-context
attribute the snapshot doesn't carry — so instead of only asserting
`document` specifically, this statically introspects every PARALLEL_AGENTS
agent's source for `context.<attr>` access and asserts every such attribute
actually exists on ContextSnapshot. A future agent added to PARALLEL_AGENTS
that reads a not-yet-snapshotted attribute fails this test immediately,
without needing anyone to remember to update an explicit list here.
"""
import ast
import inspect
from pathlib import Path

import pandas as pd
import pytest

from agents.context import AnalysisContext, ContextSnapshot
from agents.pipeline import PARALLEL_AGENTS, AGENT_REGISTRY

ROOT = Path(__file__).resolve().parent.parent


def _accessed_context_attributes(source: str) -> set[str]:
    """Static-analysis: every `<name>.<attr>` access in the source where
    `<name>` is the agent's context parameter name (conventionally
    `context`, per every agent's `_run(self, context: AnalysisContext)`
    signature)."""
    tree = ast.parse(source)
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "context":
                attrs.add(node.attr)
    return attrs


def test_every_parallel_agent_only_reads_attributes_the_snapshot_actually_has():
    snapshot_fields = {f for f in ContextSnapshot.__dataclass_fields__}
    # Methods/properties are legitimately callable on a snapshot too, not
    # just dataclass fields.
    snapshot_members = snapshot_fields | {
        name for name, _ in inspect.getmembers(ContextSnapshot, predicate=inspect.isfunction)
    }

    missing: dict[str, set[str]] = {}
    for name in PARALLEL_AGENTS:
        agent_cls = AGENT_REGISTRY[name]
        source = inspect.getsource(agent_cls)
        accessed = _accessed_context_attributes(source)
        gap = accessed - snapshot_members
        if gap:
            missing[name] = gap

    assert not missing, (
        f"parallel agents access ContextSnapshot attributes that don't exist: {missing}. "
        "Either add the field to ContextSnapshot and copy it in "
        "AnalysisContext.freeze(), or fix the agent to not need it from the snapshot."
    )


def test_document_field_present_and_copied_by_freeze():
    """The exact reported bug: context.document must survive freeze()."""
    assert "document" in ContextSnapshot.__dataclass_fields__

    ctx = AnalysisContext(df=pd.DataFrame({"a": [1]}), document={"pages": ["hello"]})
    snap = ctx.freeze()
    assert snap.document == {"pages": ["hello"]}


def test_nlp_and_vision_do_not_crash_on_a_snapshot_without_a_document():
    """document=None (the common case — no uploaded document) must be
    handled gracefully (skip with a reason), not raise AttributeError."""
    from agents.nlp_agent import NLPAgent
    from agents.vision_agent import VisionAgent

    df = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=30), "amount": range(30)})
    ctx = AnalysisContext(df=df, date_col="date", kpi_col="amount")
    snap = ctx.freeze()
    assert snap.document is None

    for agent_cls in (NLPAgent, VisionAgent):
        result = agent_cls().run(snap)
        assert result.status != "error", f"{agent_cls.__name__} crashed: {result.error}"
