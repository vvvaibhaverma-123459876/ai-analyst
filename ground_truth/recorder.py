"""
ground_truth/recorder.py
Foundation for every learning mechanism in the system.

Every finding published gets a slot here.
Without ground truth, no learning agent, calibration store,
or score tracker can improve. This is the bedrock.

Schema:
  run_id          — UUID of the analysis run
  finding_type    — anomaly | forecast | hypothesis | cluster | experiment | root_cause
  finding_id      — unique ID within the run
  finding_summary — plain-text description of the finding
  finding_data    — JSON blob of the structured finding
  agent           — which agent produced it
  method          — which juror/method within that agent
  confidence      — stated confidence at time of publication (0-1)
  outcome_actual  — what actually happened (free text + structured)
  outcome_correct — True / False / None (None = not yet verified)
  outcome_recorded_at — when verification was added
  verified_by     — user id or "auto" (system-verified from later data)
  created_at      — when the finding was published
"""

from __future__ import annotations
import sqlite3
import json
import uuid
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "ground_truth.db"


@dataclass
class Finding:
    finding_type: str
    finding_summary: str
    finding_data: dict
    agent: str
    method: str
    confidence: float
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Outcome:
    finding_id: str
    outcome_actual: str
    outcome_correct: bool | None
    verified_by: str = "user"


class GroundTruthRecorder:
    """
    The single source of truth for whether the system was right.
    Every learning agent reads from this table.
    """

    def __init__(self, db_path: str = None):
        self._db = db_path or str(DB_PATH)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS findings (
                    finding_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    finding_type TEXT,
                    finding_summary TEXT,
                    finding_data TEXT,
                    agent TEXT,
                    method TEXT,
                    confidence REAL,
                    outcome_actual TEXT,
                    outcome_correct INTEGER,
                    outcome_recorded_at TEXT,
                    verified_by TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS run_metadata (
                    run_id TEXT PRIMARY KEY,
                    data_signature TEXT,
                    agent_plan TEXT,
                    business_context TEXT,
                    data_rows INTEGER,
                    kpi_col TEXT,
                    overall_quality INTEGER,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_findings_agent
                    ON findings(agent);
                CREATE INDEX IF NOT EXISTS idx_findings_type
                    ON findings(finding_type);
                CREATE INDEX IF NOT EXISTS idx_findings_correct
                    ON findings(outcome_correct);
            """)
            conn.commit()
        logger.info("GroundTruthRecorder initialised.")

    # ------------------------------------------------------------------
    # Recording findings
    # ------------------------------------------------------------------

    def record_finding(self, finding: Finding) -> str:
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO findings
                (finding_id, run_id, finding_type, finding_summary, finding_data,
                 agent, method, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                finding.finding_id,
                finding.run_id,
                finding.finding_type,
                finding.finding_summary,
                json.dumps(finding.finding_data, default=str),
                finding.agent,
                finding.method,
                finding.confidence,
                datetime.now().isoformat(),
            ))
            conn.commit()
        return finding.finding_id

    def record_run(
        self,
        run_id: str,
        data_signature: str,
        agent_plan: list,
        business_context: dict,
        data_rows: int,
        kpi_col: str,
    ):
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO run_metadata
                (run_id, data_signature, agent_plan, business_context,
                 data_rows, kpi_col, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                data_signature,
                json.dumps(agent_plan),
                json.dumps(business_context, default=str),
                data_rows,
                kpi_col,
                datetime.now().isoformat(),
            ))
            conn.commit()

    # ------------------------------------------------------------------
    # Recording outcomes (verification)
    # ------------------------------------------------------------------

    def record_outcome(self, outcome: Outcome):
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                UPDATE findings SET
                    outcome_actual = ?,
                    outcome_correct = ?,
                    outcome_recorded_at = ?,
                    verified_by = ?
                WHERE finding_id = ?
            """, (
                outcome.outcome_actual,
                1 if outcome.outcome_correct else (0 if outcome.outcome_correct is False else None),
                datetime.now().isoformat(),
                outcome.verified_by,
                outcome.finding_id,
            ))
            conn.commit()
        logger.info(f"Outcome recorded for finding {outcome.finding_id}: {outcome.outcome_correct}")

    def bulk_verify(self, finding_ids: list[str], correct: bool, verified_by: str = "auto"):
        """Mark multiple findings as correct/incorrect at once."""
        with sqlite3.connect(self._db) as conn:
            for fid in finding_ids:
                conn.execute("""
                    UPDATE findings SET
                        outcome_correct = ?,
                        outcome_recorded_at = ?,
                        verified_by = ?
                    WHERE finding_id = ?
                """, (1 if correct else 0, datetime.now().isoformat(), verified_by, fid))
            conn.commit()

    # ------------------------------------------------------------------
    # Query interface for learning agents
    # ------------------------------------------------------------------

    def agent_accuracy(self, agent: str, method: str = None, days: int = 90) -> dict:
        """
        Returns precision, recall, and calibration for an agent/method.
        Only counts findings with verified outcomes.
        """
        with sqlite3.connect(self._db) as conn:
            q = """
                SELECT confidence, outcome_correct
                FROM findings
                WHERE agent = ?
                  AND outcome_correct IS NOT NULL
                  AND created_at >= datetime('now', ? || ' days')
            """
            params = [agent, f"-{days}"]
            if method:
                q += " AND method = ?"
                params.append(method)
            rows = conn.execute(q, params).fetchall()

        if not rows:
            return {"n": 0, "precision": None, "calibration_error": None}

        correct = [r[1] for r in rows]
        confidences = [r[0] for r in rows]
        n = len(rows)
        precision = sum(correct) / n

        # Expected calibration error (ECE)
        ece = sum(abs(c - p) for c, p in zip(confidences, correct)) / n

        return {
            "n": n,
            "precision": round(precision, 3),
            "calibration_error": round(ece, 3),
            "agent": agent,
            "method": method,
        }

    def pending_verification(self, n: int = 20) -> list[dict]:
        """Returns findings that have no outcome recorded yet."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT finding_id, finding_type, finding_summary,
                       agent, confidence, created_at
                FROM findings
                WHERE outcome_correct IS NULL
                ORDER BY created_at DESC LIMIT ?
            """, (n,)).fetchall()
        return [
            {
                "finding_id": r[0], "finding_type": r[1],
                "finding_summary": r[2], "agent": r[3],
                "confidence": r[4], "created_at": r[5],
            }
            for r in rows
        ]

    def contradiction_check(self, data_signature: str, finding_type: str) -> list[dict]:
        """
        Returns past findings of the same type on the same data signature.
        Used by Guardian to detect unstable findings across runs.
        """
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT f.finding_id, f.finding_summary, f.confidence,
                       f.outcome_correct, f.created_at
                FROM findings f
                JOIN run_metadata r ON f.run_id = r.run_id
                WHERE r.data_signature = ?
                  AND f.finding_type = ?
                ORDER BY f.created_at DESC LIMIT 10
            """, (data_signature, finding_type)).fetchall()
        return [
            {
                "finding_id": r[0], "summary": r[1],
                "confidence": r[2], "correct": r[3], "at": r[4],
            }
            for r in rows
        ]

    def update_run_quality(self, run_id: str, quality: int):
        """Quality: 1-5 integer, set post-analysis by user or guardian."""
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "UPDATE run_metadata SET overall_quality = ? WHERE run_id = ?",
                (quality, run_id)
            )
            conn.commit()
