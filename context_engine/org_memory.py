"""
context_engine/org_memory.py
Persistent org-level memory store.
Stores: company info, KPI definitions, segment definitions,
prior analyses, user corrections, learned patterns.
Backed by SQLite. Survives restarts.
"""

from __future__ import annotations
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "org_memory.db"


class OrgMemory:

    def __init__(self, db_path: str = None):
        self._db = db_path or str(DB_PATH)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS business_context (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS kpi_definitions (
                    name TEXT PRIMARY KEY,
                    definition TEXT,
                    formula TEXT,
                    owner TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS analysis_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_signature TEXT,
                    agent_plan TEXT,
                    outcome_quality INTEGER,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS user_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original TEXT,
                    correction TEXT,
                    context TEXT,
                    created_at TEXT
                );
                CREATE TABLE IF NOT EXISTS insight_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kpi TEXT,
                    finding TEXT,
                    date_range TEXT,
                    created_at TEXT
                );
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Business context (company, industry, audience, goals)
    # ------------------------------------------------------------------

    def set(self, key: str, value):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO business_context VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat())
            )
            conn.commit()

    def get(self, key: str, default=None):
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT value FROM business_context WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else default

    def get_all_context(self) -> dict:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT key, value FROM business_context"
            ).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    # ------------------------------------------------------------------
    # KPI definitions
    # ------------------------------------------------------------------

    def save_kpi(self, name: str, definition: str, formula: str = "", owner: str = ""):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO kpi_definitions VALUES (?, ?, ?, ?, ?)",
                (name, definition, formula, owner, datetime.now().isoformat())
            )
            conn.commit()

    def get_kpi(self, name: str) -> dict | None:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT name, definition, formula, owner FROM kpi_definitions WHERE name = ?",
                (name,)
            ).fetchone()
        if row:
            return {"name": row[0], "definition": row[1], "formula": row[2], "owner": row[3]}
        return None

    def all_kpis(self) -> list[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT name, definition, formula FROM kpi_definitions"
            ).fetchall()
        return [{"name": r[0], "definition": r[1], "formula": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Analysis patterns (what worked well for what data shape)
    # ------------------------------------------------------------------

    def save_pattern(self, data_signature: str, agent_plan: list, outcome_quality: int = 3):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO analysis_patterns VALUES (NULL, ?, ?, ?, ?)",
                (data_signature, json.dumps(agent_plan), outcome_quality,
                 datetime.now().isoformat())
            )
            conn.commit()

    def best_plan_for(self, data_signature: str) -> list | None:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                """SELECT agent_plan FROM analysis_patterns
                   WHERE data_signature = ? AND outcome_quality >= 4
                   ORDER BY outcome_quality DESC, id DESC LIMIT 1""",
                (data_signature,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    # ------------------------------------------------------------------
    # User corrections (feedback loop)
    # ------------------------------------------------------------------

    def save_correction(self, original: str, correction: str, context: str = ""):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO user_corrections VALUES (NULL, ?, ?, ?, ?)",
                (original, correction, context, datetime.now().isoformat())
            )
            conn.commit()

    def recent_corrections(self, n: int = 10) -> list[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT original, correction, context FROM user_corrections "
                "ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [{"original": r[0], "correction": r[1], "context": r[2]} for r in rows]

    # ------------------------------------------------------------------
    # Insight history (what has been found before)
    # ------------------------------------------------------------------

    def save_insight(self, kpi: str, finding: str, date_range: str = ""):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO insight_history VALUES (NULL, ?, ?, ?, ?)",
                (kpi, finding, date_range, datetime.now().isoformat())
            )
            conn.commit()

    def prior_insights(self, kpi: str, n: int = 5) -> list[str]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT finding FROM insight_history WHERE kpi = ? "
                "ORDER BY id DESC LIMIT ?", (kpi, n)
            ).fetchall()
        return [r[0] for r in rows]

    def to_prompt_context(self) -> str:
        """Serialise org memory for LLM prompt injection."""
        ctx = self.get_all_context()
        kpis = self.all_kpis()
        lines = ["=== Org Context ==="]
        for k, v in ctx.items():
            lines.append(f"{k}: {v}")
        if kpis:
            lines.append("\n=== KPI Definitions ===")
            for kpi in kpis:
                lines.append(f"{kpi['name']}: {kpi['definition']} | formula: {kpi['formula']}")
        return "\n".join(lines)

    def clear(self):
        with sqlite3.connect(self._db) as conn:
            for table in ["business_context", "kpi_definitions",
                          "analysis_patterns", "user_corrections", "insight_history"]:
                conn.execute(f"DELETE FROM {table}")
            conn.commit()
