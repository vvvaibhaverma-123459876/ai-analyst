from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from datetime import datetime

@dataclass
class AgentPerformance:
    agent: str
    score: float | None
    runs: int

class AgentScoreboard:
    def __init__(self, db_path: str | None = None):
        self._db = db_path or str(Path(__file__).resolve().parent.parent / 'memory' / 'guardian.db')
        Path(self._db).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS agent_runtime_scores (id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, score REAL, run_id TEXT, created_at TEXT)")
            conn.commit()
    def record(self, agent: str, score: float, run_id: str = '') -> None:
        with sqlite3.connect(self._db) as conn:
            conn.execute('INSERT INTO agent_runtime_scores (agent, score, run_id, created_at) VALUES (?, ?, ?, ?)', (agent, score, run_id, datetime.utcnow().isoformat()))
            conn.commit()
    def summary(self, agent: str) -> AgentPerformance:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute('SELECT score FROM agent_runtime_scores WHERE agent = ? ORDER BY id DESC LIMIT 100', (agent,)).fetchall()
        if not rows:
            return AgentPerformance(agent, None, 0)
        vals = [r[0] for r in rows if r[0] is not None]
        return AgentPerformance(agent, round(sum(vals) / len(vals), 3) if vals else None, len(rows))
