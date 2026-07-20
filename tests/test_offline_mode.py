"""
tests/test_offline_mode.py

Verifies the public demo's core promise: with ANALYST_MODE=offline (the
default), the governed pipeline runs end-to-end against local data and makes
*zero* outbound network calls — no LLM API, no web enrichment, nothing.

Two independent checks:
  1. Unit-level: core.config.Config blanks OPENAI_API_KEY/ANTHROPIC_API_KEY
     whenever ANALYST_MODE != "full", regardless of what's actually in the
     environment (defense in depth against a stray platform secret). Uses
     importlib.reload on core.config directly — note this does NOT propagate
     to modules that already did `from core.config import config` earlier in
     the process (Python doesn't rebind existing references on reload), so
     this only proves the blanking *logic* in core/config.py is correct.
  2. End-to-end: the actual governed pipeline, run in this process's natural
     (unmodified) environment — which in CI/Streamlit Cloud has no API keys
     set, so ANALYST_MODE's default already applies without any reload
     trickery — must complete successfully while socket.socket.connect is
     patched to raise. This is the real "zero network calls" guarantee: any
     code path anywhere in the pipeline that tries to open a connection
     fails the test loudly instead of silently succeeding on a machine that
     happens to have network access.
"""
import socket

import pytest

from core.config import config
from tests.conftest import make_ts


def test_offline_is_the_default():
    assert config.ANALYST_MODE == "offline"
    assert config.OPENAI_API_KEY == ""
    assert config.ANTHROPIC_API_KEY == ""


def test_offline_mode_blanks_keys_even_if_present_in_env(monkeypatch):
    """Defense in depth: a stray platform secret (e.g. left in Streamlit Cloud
    settings) must not re-enable paid calls unless ANALYST_MODE=full too."""
    monkeypatch.setenv("ANALYST_MODE", "offline")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-be-ignored")
    import sys
    import importlib
    # core/__init__.py does `from .config import config`, which shadows the
    # `core.config` *attribute* with the Config instance (not the module) for
    # any later `import core.config`. Go through sys.modules directly to get
    # the real module object reload() needs.
    config_module = sys.modules["core.config"]
    importlib.reload(config_module)
    try:
        assert config_module.config.OPENAI_API_KEY == ""
    finally:
        importlib.reload(config_module)  # restore the real singleton other modules hold a reference to


def test_governed_pipeline_runs_offline_with_zero_network_calls(monkeypatch):
    def _blocked_connect(self, *args, **kwargs):
        raise AssertionError(f"offline mode must not open network connections; blocked connect({args!r})")

    monkeypatch.setattr(socket.socket, "connect", _blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked_connect)

    from agents.context import AnalysisContext
    from agents.pipeline import GovernedPipeline

    df = make_ts(n=60, spike_idx=40)
    ctx = AnalysisContext(
        df=df, kpi_col="revenue", date_col="date",
        tenant_id="default", user_id="offline-smoke-test",
        business_context={"company": "Acme", "industry": "SaaS"},
    )
    finished = GovernedPipeline().run(ctx)

    assert finished.run_id
    assert finished.final_brief, "offline mode must still produce a real brief, not a blank/error state"
    assert "guardian" in finished.results
    assert "insight" in finished.results
