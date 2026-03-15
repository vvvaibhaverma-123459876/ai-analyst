"""
memory/history_store.py
Stores analysis session history in a local SQLite database.
Each entry: question, intent, SQL used, result summary, timestamp.
"""

import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "history.db"


class HistoryStore:

    def __init__(self, db_path: str = None):
        self._db = db_path or str(DB_PATH)
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    question TEXT,
                    intent TEXT,
                    sql_used TEXT,
                    kpi TEXT,
                    delta REAL,
                    pct_change REAL,
                    summary TEXT,
                    followup_questions TEXT
                )
            """)
            conn.commit()
        logger.info(f"HistoryStore initialised at {self._db}")

    def save(
        self,
        question: str,
        intent: dict = None,
        sql_used: str = None,
        kpi: str = None,
        delta: float = None,
        pct_change: float = None,
        summary: str = None,
        followup_questions: list = None,
    ):
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT INTO analysis_history
                (timestamp, question, intent, sql_used, kpi, delta, pct_change, summary, followup_questions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                question,
                json.dumps(intent or {}),
                sql_used or "",
                kpi or "",
                delta,
                pct_change,
                summary or "",
                json.dumps(followup_questions or []),
            ))
            conn.commit()
        logger.info(f"Session saved: '{question[:60]}'")

    def recent(self, n: int = 10) -> list[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT timestamp, question, kpi, delta, pct_change, summary "
                "FROM analysis_history ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [
            {
                "timestamp": r[0],
                "question": r[1],
                "kpi": r[2],
                "delta": r[3],
                "pct_change": r[4],
                "summary": r[5],
            }
            for r in rows
        ]

    def learned_questions(self, n: int = 20) -> list[str]:
        """Returns past questions — useful for autocomplete / suggestions."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT DISTINCT question FROM analysis_history ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [r[0] for r in rows]

    def clear(self):
        with sqlite3.connect(self._db) as conn:
            conn.execute("DELETE FROM analysis_history")
            conn.commit()
        logger.info("History cleared.")
