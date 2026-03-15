"""
context_engine/org_memory.py  — v0.6
Persistent org-level memory store.

Upgrade vs v0.5:
  - Dual-write: SQLite (audit/export) + VectorStore (semantic retrieval)
  - semantic_prior_insights()  — returns semantically similar past findings
  - semantic_search_context()  — free-text search across all org knowledge
  - All write operations also index into the vector store
  - Falls back gracefully to keyword search if vector store unavailable
"""

from __future__ import annotations
import sqlite3
import json
import hashlib
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

    # ------------------------------------------------------------------
    # Vector store (lazy init — never blocks startup)
    # ------------------------------------------------------------------

    def _vs(self):
        if not hasattr(self, "_vector_store"):
            try:
                from memory.vector_store import vector_store
                self._vector_store = vector_store()
            except Exception as e:
                logger.warning(f"VectorStore unavailable: {e}")
                self._vector_store = None
        return self._vector_store

    # ------------------------------------------------------------------
    # Semantic retrieval
    # ------------------------------------------------------------------

    def semantic_prior_insights(self, query: str, kpi: str = "",
                                n: int = 5, min_score: float = 0.3) -> list[dict]:
        """
        Returns semantically similar past findings for a query.
        Each result: {"text": ..., "kpi": ..., "score": ..., "date": ...}
        Falls back to keyword prior_insights() if vector store is unavailable.
        """
        vs = self._vs()
        if vs is None:
            # Graceful fallback
            return [{"text": t, "kpi": kpi, "score": 0.5, "date": ""}
                    for t in self.prior_insights(kpi, n)]
        where = {"kpi": {"$eq": kpi}} if kpi else None
        results = vs.query("insights", query, n=n, where=where) if where else \
                  vs.query("insights", query, n=n)
        return [
            {
                "text": r["text"],
                "kpi": r["metadata"].get("kpi", ""),
                "score": r["score"],
                "date": r["metadata"].get("date", ""),
            }
            for r in results if r["score"] >= min_score
        ]

    def semantic_search_context(self, query: str, n: int = 8) -> list[dict]:
        """
        Free-text semantic search across ALL org knowledge collections.
        Returns merged, ranked results from insights + corrections + patterns + context.
        """
        vs = self._vs()
        if vs is None:
            return []
        all_results = []
        for col in ["insights", "corrections", "patterns", "context"]:
            hits = vs.query(col, query, n=max(n // 4, 2))
            for h in hits:
                h["collection"] = col
            all_results.extend(hits)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:n]

    def to_semantic_prompt_context(self, query: str, max_items: int = 6) -> str:
        """
        Build a rich LLM prompt context using semantic retrieval instead of
        full-table dumps. Surfaces the most relevant org knowledge for the query.
        """
        results = self.semantic_search_context(query, n=max_items)
        if not results:
            return self.to_prompt_context()     # fallback to original

        lines = ["=== Semantically Retrieved Org Context ==="]
        for r in results:
            col = r.get("collection", "?")
            score = r.get("score", 0)
            lines.append(f"[{col} | relevance={score:.2f}] {r['text']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Override write methods to dual-write into vector store
    # ------------------------------------------------------------------

    def save_insight(self, kpi: str, finding: str, date_range: str = ""):
        # SQLite write (original)
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO insight_history VALUES (NULL, ?, ?, ?, ?)",
                (kpi, finding, date_range, datetime.now().isoformat())
            )
            conn.commit()
        # Vector write
        vs = self._vs()
        if vs:
            doc_id = hashlib.md5(f"{kpi}:{finding}".encode()).hexdigest()
            vs.upsert("insights", doc_id, finding,
                      metadata={"kpi": kpi, "date": date_range})

    def save_correction(self, original: str, correction: str, context: str = ""):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO user_corrections VALUES (NULL, ?, ?, ?, ?)",
                (original, correction, context, datetime.now().isoformat())
            )
            conn.commit()
        vs = self._vs()
        if vs:
            doc_id = hashlib.md5(f"{original}:{correction}".encode()).hexdigest()
            vs.upsert("corrections", doc_id,
                      f"Original: {original}\nCorrection: {correction}",
                      metadata={"context": context})

    def save_pattern(self, data_signature: str, agent_plan: list, outcome_quality: int = 3):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO analysis_patterns VALUES (NULL, ?, ?, ?, ?)",
                (data_signature, json.dumps(agent_plan), outcome_quality,
                 datetime.now().isoformat())
            )
            conn.commit()
        vs = self._vs()
        if vs:
            doc_id = hashlib.md5(data_signature.encode()).hexdigest()
            vs.upsert("patterns", doc_id,
                      f"Data pattern: {data_signature}\nPlan: {agent_plan}",
                      metadata={"signature": data_signature, "quality": outcome_quality})

    def set(self, key: str, value):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO business_context VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.now().isoformat())
            )
            conn.commit()
        vs = self._vs()
        if vs:
            vs.upsert("context", key, f"{key}: {json.dumps(value)}",
                      metadata={"key": key})

    def clear(self):
        with sqlite3.connect(self._db) as conn:
            for table in ["business_context", "kpi_definitions",
                          "analysis_patterns", "user_corrections", "insight_history"]:
                conn.execute(f"DELETE FROM {table}")
            conn.commit()
