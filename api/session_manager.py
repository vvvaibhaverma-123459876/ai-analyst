"""
api/session_manager.py  — v0.6
Multi-user session manager.

Bridges the RBAC/JWT API layer with the Streamlit UI and OrgMemory.
Provides:
  - User identity resolution from JWT or session token
  - Per-user analysis history namespaced in OrgMemory
  - Shared workspace (team annotations visible to all team members)
  - Finding annotation: any user can annotate / dispute a finding
  - Session-level state (current connector, active KPI, grain)

This makes the system genuinely multi-user without requiring a full
rewrite of the single-user Streamlit app — the session manager is
injected into the app's AnalysisContext as context.user_id and
context.tenant_id, and all OrgMemory writes are namespaced.
"""

from __future__ import annotations
import sqlite3
import json
import hashlib
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

logger = get_logger(__name__)

SESSION_DB = Path(__file__).resolve().parent.parent / "memory" / "sessions.db"


# ------------------------------------------------------------------
# Data models
# ------------------------------------------------------------------

@dataclass
class UserSession:
    session_id: str
    user_id: str
    tenant_id: str
    role: str                   # viewer | analyst | admin
    display_name: str
    email: str
    created_at: str
    last_active: str
    preferences: dict = field(default_factory=dict)
    # Active state (not persisted — reconstructed each load)
    active_connector: str = ""
    active_kpi: str = ""
    active_grain: str = "Daily"


@dataclass
class FindingAnnotation:
    annotation_id: str
    run_id: str
    finding_key: str            # "anomaly" | "hypothesis:uuid" | "root_cause"
    user_id: str
    tenant_id: str
    verdict: str                # "correct" | "incorrect" | "disputed" | "note"
    comment: str
    created_at: str


# ------------------------------------------------------------------
# Session DB
# ------------------------------------------------------------------

class SessionStore:

    def __init__(self, db_path: str = None):
        self._db = db_path or str(SESSION_DB)
        os.makedirs(os.path.dirname(self._db), exist_ok=True)
        self._init()

    def _init(self):
        with sqlite3.connect(self._db) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id   TEXT PRIMARY KEY,
                    user_id      TEXT NOT NULL,
                    tenant_id    TEXT NOT NULL DEFAULT 'default',
                    role         TEXT NOT NULL DEFAULT 'analyst',
                    display_name TEXT,
                    email        TEXT,
                    created_at   TEXT,
                    last_active  TEXT,
                    preferences  TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id TEXT PRIMARY KEY,
                    run_id        TEXT,
                    finding_key   TEXT,
                    user_id       TEXT,
                    tenant_id     TEXT,
                    verdict       TEXT,
                    comment       TEXT,
                    created_at    TEXT
                );
                CREATE TABLE IF NOT EXISTS shared_workspace (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id   TEXT NOT NULL,
                    user_id     TEXT,
                    key         TEXT,
                    value       TEXT,
                    created_at  TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_annotations_run
                    ON annotations(run_id, tenant_id);
                CREATE INDEX IF NOT EXISTS idx_workspace_tenant
                    ON shared_workspace(tenant_id, key);
            """)
            conn.commit()

    # ------------------------------------------------------------------
    # Session CRUD
    # ------------------------------------------------------------------

    def create_session(
        self,
        user_id: str,
        tenant_id: str = "default",
        role: str = "analyst",
        display_name: str = "",
        email: str = "",
        preferences: dict = None,
    ) -> UserSession:
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)",
                (session_id, user_id, tenant_id, role,
                 display_name or user_id, email, now, now,
                 json.dumps(preferences or {}))
            )
            conn.commit()
        return UserSession(
            session_id=session_id, user_id=user_id, tenant_id=tenant_id,
            role=role, display_name=display_name or user_id, email=email,
            created_at=now, last_active=now, preferences=preferences or {},
        )

    def get_session(self, session_id: str) -> UserSession | None:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
        if not row:
            return None
        self._touch(session_id)
        return UserSession(
            session_id=row[0], user_id=row[1], tenant_id=row[2],
            role=row[3], display_name=row[4], email=row[5],
            created_at=row[6], last_active=row[7],
            preferences=json.loads(row[8] or "{}"),
        )

    def get_or_create(
        self, user_id: str, tenant_id: str = "default", **kwargs
    ) -> UserSession:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT session_id FROM sessions WHERE user_id=? AND tenant_id=? "
                "ORDER BY last_active DESC LIMIT 1",
                (user_id, tenant_id)
            ).fetchone()
        if row:
            session = self.get_session(row[0])
            if session:
                return session
        return self.create_session(user_id, tenant_id, **kwargs)

    def _touch(self, session_id: str):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "UPDATE sessions SET last_active=? WHERE session_id=?",
                (datetime.now().isoformat(), session_id)
            )
            conn.commit()

    def list_sessions(self, tenant_id: str = "default") -> list[UserSession]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE tenant_id=? ORDER BY last_active DESC",
                (tenant_id,)
            ).fetchall()
        return [
            UserSession(
                session_id=r[0], user_id=r[1], tenant_id=r[2],
                role=r[3], display_name=r[4], email=r[5],
                created_at=r[6], last_active=r[7],
                preferences=json.loads(r[8] or "{}"),
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def annotate(
        self,
        run_id: str,
        finding_key: str,
        user_id: str,
        tenant_id: str,
        verdict: str,
        comment: str = "",
    ) -> FindingAnnotation:
        annotation_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO annotations VALUES (?,?,?,?,?,?,?,?)",
                (annotation_id, run_id, finding_key,
                 user_id, tenant_id, verdict, comment, now)
            )
            conn.commit()
        # Also push to ground truth recorder
        try:
            from ground_truth.recorder import GroundTruthRecorder, Finding, Outcome
            recorder = GroundTruthRecorder()
            outcome = {
                "correct": Outcome.CORRECT,
                "incorrect": Outcome.INCORRECT,
                "disputed": Outcome.INCORRECT,
                "note": Outcome.PENDING,
            }.get(verdict, Outcome.PENDING)
            recorder.record_outcome(run_id=run_id, finding_key=finding_key, outcome=outcome)
        except Exception as e:
            logger.warning("Ground truth recording failed: %s", e)

        return FindingAnnotation(
            annotation_id=annotation_id, run_id=run_id,
            finding_key=finding_key, user_id=user_id,
            tenant_id=tenant_id, verdict=verdict,
            comment=comment, created_at=now,
        )

    def get_annotations(
        self, run_id: str, tenant_id: str = "default"
    ) -> list[FindingAnnotation]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE run_id=? AND tenant_id=? "
                "ORDER BY created_at DESC",
                (run_id, tenant_id)
            ).fetchall()
        return [
            FindingAnnotation(
                annotation_id=r[0], run_id=r[1], finding_key=r[2],
                user_id=r[3], tenant_id=r[4], verdict=r[5],
                comment=r[6], created_at=r[7],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Shared workspace
    # ------------------------------------------------------------------

    def workspace_set(self, tenant_id: str, user_id: str, key: str, value: Any):
        with sqlite3.connect(self._db) as conn:
            conn.execute(
                "INSERT INTO shared_workspace(tenant_id,user_id,key,value,created_at) "
                "VALUES (?,?,?,?,?)",
                (tenant_id, user_id, key, json.dumps(value), datetime.now().isoformat())
            )
            conn.commit()

    def workspace_get(self, tenant_id: str, key: str) -> Any:
        with sqlite3.connect(self._db) as conn:
            row = conn.execute(
                "SELECT value FROM shared_workspace WHERE tenant_id=? AND key=? "
                "ORDER BY id DESC LIMIT 1",
                (tenant_id, key)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def workspace_list(self, tenant_id: str, prefix: str = "") -> list[dict]:
        with sqlite3.connect(self._db) as conn:
            rows = conn.execute(
                "SELECT key, value, user_id, created_at FROM shared_workspace "
                "WHERE tenant_id=? AND key LIKE ? ORDER BY id DESC",
                (tenant_id, f"{prefix}%")
            ).fetchall()
        return [
            {"key": r[0], "value": json.loads(r[1]), "user_id": r[2], "created_at": r[3]}
            for r in rows
        ]


# ------------------------------------------------------------------
# Streamlit helper (inject into AnalysisContext)
# ------------------------------------------------------------------

def streamlit_session(
    tenant_id: str = "default",
    role: str = "analyst",
    display_name: str = "",
) -> UserSession:
    """
    Returns or creates a UserSession tied to the Streamlit session.
    Call at the top of v06_app.py to wire multi-user identity.
    """
    try:
        import streamlit as st
        store = SessionStore()
        key = "ai_analyst_session_id"
        if key not in st.session_state:
            # Create new session
            user_id = f"st_user_{uuid.uuid4().hex[:8]}"
            session = store.create_session(
                user_id=user_id,
                tenant_id=tenant_id,
                role=role,
                display_name=display_name or f"User {user_id[-4:]}",
            )
            st.session_state[key] = session.session_id
            return session
        session = store.get_session(st.session_state[key])
        if session is None:
            # Stale session ID — create fresh
            del st.session_state[key]
            return streamlit_session(tenant_id, role, display_name)
        return session
    except ImportError:
        # Not in Streamlit — return anonymous session
        return UserSession(
            session_id=str(uuid.uuid4()), user_id="anonymous",
            tenant_id=tenant_id, role=role,
            display_name="Anonymous", email="",
            created_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
        )
