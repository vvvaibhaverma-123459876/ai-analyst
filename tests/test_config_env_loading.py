"""
tests/test_config_env_loading.py

Reproduces and locks in the fix for the observed bug: the app ran in offline
mode despite a real .env file at the repo root setting ANALYST_MODE=full.
core/config.py now points python-dotenv at BASE_DIR / ".env" explicitly
instead of relying on its frame-based implicit discovery.

These tests write a REAL .env file at the actual repo root (core.config.BASE_DIR)
because that's literally what's under test — "loads .env from the repo root" —
and reload the real core.config module (via sys.modules, working around the
core/__init__.py `from .config import config` shadowing quirk documented in
tests/test_offline_mode.py). Any pre-existing .env is backed up and restored
so this never clobbers a developer's real local file.
"""
import os
import sys
import importlib

import pytest

from core.config import BASE_DIR


@pytest.fixture
def repo_root_env_file():
    """Yields a function to write repo-root .env content; backs up/restores
    any pre-existing .env and always reloads core.config back to its
    pre-test state afterward."""
    env_path = BASE_DIR / ".env"
    backup = env_path.read_text(encoding="utf-8") if env_path.exists() else None

    def _write(content: str):
        env_path.write_text(content, encoding="utf-8")

    try:
        yield _write
    finally:
        if backup is not None:
            env_path.write_text(backup, encoding="utf-8")
        elif env_path.exists():
            env_path.unlink()
        # Restore process env + reload the singleton to the pre-test state.
        for var in ("ANALYST_MODE", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
            os.environ.pop(var, None)
        importlib.reload(sys.modules["core.config"])


def test_env_file_at_repo_root_sets_full_mode(repo_root_env_file, monkeypatch):
    monkeypatch.delenv("ANALYST_MODE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    repo_root_env_file(
        "ANALYST_MODE=full\n"
        "OPENAI_API_KEY=sk-test-not-real\n"
        "LLM_PROVIDER=openai\n"
        "LLM_MODEL=gpt-4o-mini\n"
    )

    config_module = importlib.reload(sys.modules["core.config"])

    assert config_module.config.ANALYST_MODE == "full"
    assert config_module.config.OPENAI_API_KEY == "sk-test-not-real"
    assert config_module.config.LLM_PROVIDER == "openai"
    assert config_module.config.LLM_MODEL == "gpt-4o-mini"


def test_real_environment_variable_wins_over_env_file(repo_root_env_file, monkeypatch):
    """override=False: a real already-set env var must not be clobbered by
    the .env file — matters on hosts (Streamlit Cloud, CI) that inject
    ANALYST_MODE/keys as real process env vars, not via a .env file."""
    monkeypatch.setenv("ANALYST_MODE", "offline")
    repo_root_env_file("ANALYST_MODE=full\n")

    config_module = importlib.reload(sys.modules["core.config"])

    assert config_module.config.ANALYST_MODE == "offline"


def test_key_detected_but_offline_hint(repo_root_env_file, monkeypatch):
    # load_dotenv() only ever adds to os.environ, never removes — clear the
    # relevant vars before every reload so each case starts clean.
    def _reset_env():
        for var in ("ANALYST_MODE", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(var, raising=False)

    _reset_env()
    repo_root_env_file("ANALYST_MODE=offline\nOPENAI_API_KEY=sk-test-not-real\n")
    config_module = importlib.reload(sys.modules["core.config"])
    assert config_module.config.key_detected_but_offline == "openai"

    _reset_env()
    repo_root_env_file("ANALYST_MODE=full\nOPENAI_API_KEY=sk-test-not-real\n")
    config_module = importlib.reload(sys.modules["core.config"])
    assert config_module.config.key_detected_but_offline is None

    _reset_env()
    repo_root_env_file("ANALYST_MODE=offline\n")
    config_module = importlib.reload(sys.modules["core.config"])
    assert config_module.config.key_detected_but_offline is None


def test_mode_badge_html_is_explicit(repo_root_env_file, monkeypatch):
    """The badge itself (not just the underlying config) must be explicit:
    full mode names the provider/model; offline mode surfaces the
    key-detected-but-ignored hint instead of looking identical to
    "no key configured". Needs streamlit (app/ui/product_app.py imports it
    at module level) — skipped under the lean requirements-ci.txt env,
    covered by the demo-smoke CI job which installs requirements-demo.txt."""
    pytest.importorskip("streamlit")

    def _reset_env():
        for var in ("ANALYST_MODE", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
            monkeypatch.delenv(var, raising=False)

    _reset_env()
    repo_root_env_file("ANALYST_MODE=full\nOPENAI_API_KEY=sk-test\nLLM_PROVIDER=openai\nLLM_MODEL=gpt-4o-mini\n")
    importlib.reload(sys.modules["core.config"])
    import app.ui.product_app as product_app
    importlib.reload(product_app)
    html = product_app.mode_badge_html()
    assert "Full mode" in html
    assert "openai" in html
    assert "gpt-4o-mini" in html

    _reset_env()
    repo_root_env_file("ANALYST_MODE=offline\nOPENAI_API_KEY=sk-test\n")
    importlib.reload(sys.modules["core.config"])
    importlib.reload(product_app)
    html = product_app.mode_badge_html()
    assert "Offline demo mode" in html
    assert "OPENAI_API_KEY detected but ANALYST_MODE=offline" in html

    _reset_env()
    repo_root_env_file("ANALYST_MODE=offline\n")
    importlib.reload(sys.modules["core.config"])
    importlib.reload(product_app)
    html = product_app.mode_badge_html()
    assert "Offline demo mode" in html
    assert "detected but" not in html
