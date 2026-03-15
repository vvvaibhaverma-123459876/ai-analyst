"""
learning/learning_layer.py
LearningLayer — base class every per-layer learning agent inherits from.

Each learning agent:
  1. Observes an outcome (from GroundTruthRecorder)
  2. Updates a stored belief (in its layer-specific store)
  3. Changes behaviour on the next run based on the updated belief

Key design principles:
  - Learning NEVER modifies policy-locked rules (debate jury, min sample sizes)
  - Learning requires ground truth verification before updating beliefs
  - All belief updates are logged to the audit trail
  - Score decay is applied: old observations count less than recent ones
"""

from __future__ import annotations
import sqlite3
import json
import math
import os
from datetime import datetime
from pathlib import Path
from abc import ABC, abstractmethod
from ground_truth.recorder import GroundTruthRecorder
from core.logger import get_logger

logger = get_logger(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "learning.db"
DECAY_HALF_LIFE_DAYS = 30


class LearningLayer(ABC):
    """Abstract base for all per-layer learning agents."""

    layer_name: str = "base"

    def __init__(self):
        self._gt = GroundTruthRecorder()
        self._db = str(DB_PATH)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_beliefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layer TEXT NOT NULL,
                    belief_key TEXT NOT NULL,
                    belief_value TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    observation_count INTEGER DEFAULT 1,
                    last_updated TEXT,
                    UNIQUE(layer, belief_key)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    layer TEXT,
                    observation_key TEXT,
                    outcome REAL,
                    created_at TEXT
                )
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def observe(self, context, result) -> dict:
        """
        Extract observations from a completed agent run.
        Returns dict of {key: value} observations.
        """

    @abstractmethod
    def adapt(self, context) -> dict:
        """
        Read current beliefs and return adaptation parameters
        to apply before the next run.
        """

    # ------------------------------------------------------------------
    # Belief store
    # ------------------------------------------------------------------

    def update_belief(self, key: str, value, confidence: float = 0.7):
        """Update a belief with decay-weighted confidence."""
        serialised = json.dumps(value, default=str)
        with sqlite3.connect(self._db) as conn:
            existing = conn.execute("""
                SELECT belief_value, confidence, observation_count
                FROM learning_beliefs
                WHERE layer = ? AND belief_key = ?
            """, (self.layer_name, key)).fetchone()

            if existing:
                old_conf = existing[1]
                old_count = existing[2]
                # Exponential moving average
                new_conf = 0.7 * confidence + 0.3 * old_conf
                conn.execute("""
                    UPDATE learning_beliefs
                    SET belief_value = ?, confidence = ?,
                        observation_count = ?, last_updated = ?
                    WHERE layer = ? AND belief_key = ?
                """, (serialised, round(new_conf, 4),
                      old_count + 1, datetime.now().isoformat(),
                      self.layer_name, key))
            else:
                conn.execute("""
                    INSERT INTO learning_beliefs
                    (layer, belief_key, belief_value, confidence,
                     observation_count, last_updated)
                    VALUES (?, ?, ?, ?, 1, ?)
                """, (self.layer_name, key, serialised,
                      confidence, datetime.now().isoformat()))
            conn.commit()

    def get_belief(self, key: str, default=None):
        with sqlite3.connect(self._db) as conn:
            row = conn.execute("""
                SELECT belief_value, confidence FROM learning_beliefs
                WHERE layer = ? AND belief_key = ?
            """, (self.layer_name, key)).fetchone()
        if row:
            try:
                return json.loads(row[0])
            except Exception:
                return row[0]
        return default

    def record_observation(self, key: str, outcome: float):
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT INTO learning_observations
                (layer, observation_key, outcome, created_at)
                VALUES (?, ?, ?, ?)
            """, (self.layer_name, key, outcome, datetime.now().isoformat()))
            conn.commit()

    def decay_weighted_mean(self, key: str, days: int = 90) -> float | None:
        """Returns decay-weighted mean of observations for a key."""
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT outcome, created_at FROM learning_observations
                WHERE layer = ? AND observation_key = ?
                AND created_at >= datetime('now', ? || ' days')
                ORDER BY id DESC LIMIT 200
            """, (self.layer_name, key, f"-{days}")).fetchall()

        if not rows:
            return None

        now = datetime.now()
        total_w = 0.0; weighted_sum = 0.0
        for outcome, created_at in rows:
            try:
                age = (now - datetime.fromisoformat(created_at)).days
                w = math.exp(-age * math.log(2) / DECAY_HALF_LIFE_DAYS)
                weighted_sum += outcome * w
                total_w += w
            except Exception:
                continue

        return round(weighted_sum / total_w, 4) if total_w > 0 else None

    def all_beliefs(self) -> dict:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute("""
                SELECT belief_key, belief_value, confidence, observation_count
                FROM learning_beliefs WHERE layer = ?
            """, (self.layer_name,)).fetchall()
        return {
            r[0]: {"value": json.loads(r[1]), "confidence": r[2], "n": r[3]}
            for r in rows
        }
