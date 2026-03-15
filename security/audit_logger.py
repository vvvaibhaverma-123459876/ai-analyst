"""
security/audit_logger.py
Immutable audit log for every external call, data access, and policy decision.
Append-only SQLite. No deletes. Used for compliance and debugging.

Logs:
  - every LLM API call (model, prompt hash, NOT prompt content, user, timestamp)
  - every data ingestion event (file type, row count, classification summary)
  - every policy decision (what was blocked and why)
  - every finding published (finding_id, agent, confidence)
  - every user action (verify, approve, override)
  - every external call (scraper, API enrichment)
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

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "audit.db"


class AuditLogger:

    def __init__(self, db_path: str = None):
        self._db = db_path or str(DB_PATH)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    tenant_id TEXT,
                    user_id TEXT,
                    run_id TEXT,
                    summary TEXT,
                    detail TEXT,
                    data_hash TEXT,
                    external_call INTEGER DEFAULT 0,
                    blocked INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_event
                    ON audit_log(event_type);
                CREATE INDEX IF NOT EXISTS idx_audit_run
                    ON audit_log(run_id);
                CREATE INDEX IF NOT EXISTS idx_audit_tenant
                    ON audit_log(tenant_id);
            """)
            conn.commit()

    def _write(self, event_type: str, summary: str, detail: dict = None,
               tenant_id: str = None, user_id: str = None, run_id: str = None,
               data_hash: str = None, external_call: bool = False, blocked: bool = False):
        with sqlite3.connect(self._db) as conn:
            conn.execute("""
                INSERT INTO audit_log
                (event_type, tenant_id, user_id, run_id, summary, detail,
                 data_hash, external_call, blocked, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event_type, tenant_id, user_id, run_id,
                summary, json.dumps(detail or {}, default=str),
                data_hash, 1 if external_call else 0,
                1 if blocked else 0,
                datetime.now().isoformat(),
            ))
            conn.commit()

    # ------------------------------------------------------------------
    # Public log methods
    # ------------------------------------------------------------------

    def log_llm_call(self, model: str, prompt: str, provider: str,
                     run_id: str = None, tenant_id: str = None, user_id: str = None):
        """Logs the call but NEVER the prompt content — only its hash."""
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        self._write(
            "LLM_CALL",
            summary=f"{provider}/{model}",
            detail={"model": model, "provider": provider, "prompt_hash": prompt_hash},
            run_id=run_id, tenant_id=tenant_id, user_id=user_id,
            external_call=True,
        )

    def log_ingestion(self, filename: str, file_type: str, rows: int,
                      classification_summary: str, warnings: list,
                      run_id: str = None, tenant_id: str = None):
        self._write(
            "DATA_INGESTION",
            summary=f"{filename} ({file_type}, {rows:,} rows)",
            detail={
                "filename": filename, "file_type": file_type,
                "rows": rows, "classification": classification_summary,
                "warnings": warnings,
            },
            run_id=run_id, tenant_id=tenant_id,
        )

    def log_policy_block(self, rule: str, reason: str, context: str,
                         run_id: str = None, tenant_id: str = None):
        self._write(
            "POLICY_BLOCK",
            summary=f"Blocked by policy: {rule}",
            detail={"rule": rule, "reason": reason, "context": context},
            run_id=run_id, tenant_id=tenant_id, blocked=True,
        )

    def log_finding_published(self, finding_id: str, finding_type: str,
                               agent: str, confidence: float,
                               run_id: str = None, tenant_id: str = None):
        self._write(
            "FINDING_PUBLISHED",
            summary=f"{agent}/{finding_type} (confidence={confidence:.2f})",
            detail={"finding_id": finding_id, "agent": agent,
                    "finding_type": finding_type, "confidence": confidence},
            run_id=run_id, tenant_id=tenant_id,
        )

    def log_user_action(self, action: str, target_id: str, detail: dict = None,
                        user_id: str = None, tenant_id: str = None):
        self._write(
            f"USER_{action.upper()}",
            summary=f"User {action} on {target_id}",
            detail={"target_id": target_id, **(detail or {})},
            user_id=user_id, tenant_id=tenant_id,
        )

    def log_external_call(self, service: str, url_hash: str, status: str,
                           run_id: str = None, tenant_id: str = None):
        self._write(
            "EXTERNAL_CALL",
            summary=f"{service} → {status}",
            detail={"service": service, "url_hash": url_hash, "status": status},
            run_id=run_id, tenant_id=tenant_id, external_call=True,
        )

    def log_pii_masking(self, masked_cols: list, dropped_cols: list,
                         run_id: str = None, tenant_id: str = None):
        self._write(
            "PII_MASKING",
            summary=f"Masked {len(masked_cols)} PII, dropped {len(dropped_cols)} sensitive cols",
            detail={"masked": masked_cols, "dropped": dropped_cols},
            run_id=run_id, tenant_id=tenant_id,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def recent(self, n: int = 50, tenant_id: str = None) -> list[dict]:
        with sqlite3.connect(self._db) as conn:
            q = "SELECT event_type, tenant_id, user_id, run_id, summary, created_at FROM audit_log"
            params = []
            if tenant_id:
                q += " WHERE tenant_id = ?"
                params.append(tenant_id)
            q += " ORDER BY id DESC LIMIT ?"
            params.append(n)
            rows = conn.execute(q, params).fetchall()
        return [
            {"event_type": r[0], "tenant_id": r[1], "user_id": r[2],
             "run_id": r[3], "summary": r[4], "at": r[5]}
            for r in rows
        ]

    def external_call_count(self, days: int = 30, tenant_id: str = None) -> int:
        with sqlite3.connect(self._db) as conn:
            q = """SELECT COUNT(*) FROM audit_log
                   WHERE external_call = 1
                   AND created_at >= datetime('now', ? || ' days')"""
            params = [f"-{days}"]
            if tenant_id:
                q += " AND tenant_id = ?"
                params.append(tenant_id)
            return conn.execute(q, params).fetchone()[0]
