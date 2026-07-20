"""
core/config.py
Central configuration loader. Reads .env + configs/*.yaml files.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"

# Load .env from the repo root explicitly, rather than relying on
# python-dotenv's implicit frame-walking discovery (find_dotenv() resolves
# relative to whatever frame happens to call it, which is fragile under
# Streamlit's script-execution model and easy to get wrong silently — the
# .env would just never load, with no error, and the app falls back to
# offline mode without any indication why). override=False: real
# already-set environment variables (e.g. a platform's own env/secrets)
# always win over .env file values.
load_dotenv(BASE_DIR / ".env", override=False)


def _load_yaml(filename: str) -> dict:
    path = CONFIGS_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class Config:
    # Single, explicit switch for the whole app: "offline" (default) | "full".
    # Offline is the only mode the public demo deploy should ever run in.
    #
    # Every LLM-touching module in this codebase already guards on
    # `config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY` being falsy and
    # degrades to a deterministic/rule-based path when it is (see e.g.
    # output/conversation_engine.py, agents/insight_agent.py,
    # context_engine/context_engine.py). Blanking both keys here — regardless
    # of what's actually set in the environment — makes that fallback
    # unconditional in offline mode: a stray platform secret (e.g. an
    # OPENAI_API_KEY left over in Streamlit Cloud secrets) can no longer
    # cause a paid call. Set ANALYST_MODE=full to opt back in.
    ANALYST_MODE: str = os.getenv("ANALYST_MODE", "offline").strip().lower()

    # LLM provider — switchable later: "openai" | "anthropic" | "gemini"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    # Raw env values, kept regardless of mode so the UI can tell a visitor
    # "a key IS configured but offline mode is ignoring it" instead of
    # silently looking identical to "no key configured at all".
    _RAW_OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    _RAW_ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY: str = _RAW_OPENAI_API_KEY if ANALYST_MODE == "full" else ""
    ANTHROPIC_API_KEY: str = _RAW_ANTHROPIC_API_KEY if ANALYST_MODE == "full" else ""
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    # Athena (optional)
    ATHENA_REGION: str = os.getenv("ATHENA_REGION", "ap-south-1")
    ATHENA_S3_STAGING_DIR: str = os.getenv("ATHENA_S3_STAGING_DIR", "")
    ATHENA_DATABASE: str = os.getenv("ATHENA_DATABASE", "default")

    # Analysis defaults
    DEFAULT_ROLLING_WINDOW: int = int(os.getenv("DEFAULT_ROLLING_WINDOW", "7"))
    DEFAULT_Z_THRESHOLD: float = float(os.getenv("DEFAULT_Z_THRESHOLD", "2.0"))
    DEFAULT_DRIVER_DAYS: int = int(os.getenv("DEFAULT_DRIVER_DAYS", "3"))
    MAX_QUERY_ROWS: int = int(os.getenv("MAX_QUERY_ROWS", "100000"))

    # Loaded from yaml
    METRICS: dict = _load_yaml("metrics.yaml")
    TABLES: dict = _load_yaml("tables.yaml")
    JOINS: dict = _load_yaml("joins.yaml")
    GLOSSARY: dict = _load_yaml("business_glossary.yaml")

    @property
    def key_detected_but_offline(self) -> str | None:
        """The provider name if a key was found in the raw environment but
        ANALYST_MODE isn't "full" (so it's being ignored) — None otherwise.
        UI-hint only; never affects which provider actually runs."""
        if self.ANALYST_MODE == "full":
            return None
        if self._RAW_OPENAI_API_KEY:
            return "openai"
        if self._RAW_ANTHROPIC_API_KEY:
            return "anthropic"
        return None


config = Config()
