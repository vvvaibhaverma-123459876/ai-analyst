"""
core/config.py
Central configuration loader. Reads .env + configs/*.yaml files.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIGS_DIR = BASE_DIR / "configs"


def _load_yaml(filename: str) -> dict:
    path = CONFIGS_DIR / filename
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


class Config:
    # LLM provider — switchable later: "openai" | "anthropic" | "gemini"
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
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


config = Config()
