"""api/job_store.py — Job queue backed by SQLite."""

from __future__ import annotations
import sqlite3, json, os
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "jobs.db"


class JobStatus:
    QUEUED   = "queued"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"


@dataclass
class Job:
    job_id: str
    tenant_id: str
    user_id: str
    filename: str
    file_content: bytes
    kpi_col: Optional[str]
    date_col: Optional[str]
    grain: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: dict = field(default_factory=dict)


class JobStore:

    def __init__(self):
        os.makedirs(str(DB_PATH.parent), exist_ok=True)
        self._init_db()
        self._cache: dict[str, Job] = {}

    def _init_db(self):
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT, user_id TEXT, filename TEXT,
                    kpi_col TEXT, date_col TEXT, grain TEXT,
                    status TEXT, created_at TEXT, started_at TEXT,
                    completed_at TEXT, error TEXT, result TEXT
                )
            """)
            conn.commit()

    def save(self, job: Job):
        self._cache[job.job_id] = job
        with sqlite3.connect(str(DB_PATH)) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs
                (job_id, tenant_id, user_id, filename, kpi_col, date_col, grain,
                 status, created_at, started_at, completed_at, error, result)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                job.job_id, job.tenant_id, job.user_id, job.filename,
                job.kpi_col, job.date_col, job.grain,
                job.status, job.created_at, job.started_at,
                job.completed_at, job.error, json.dumps(job.result, default=str),
            ))
            conn.commit()

    def get(self, job_id: str) -> Optional[Job]:
        if job_id in self._cache:
            return self._cache[job_id]
        with sqlite3.connect(str(DB_PATH)) as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        job = Job(
            job_id=row[0], tenant_id=row[1], user_id=row[2], filename=row[3],
            file_content=b"", kpi_col=row[4], date_col=row[5], grain=row[6],
            status=row[7], created_at=row[8], started_at=row[9],
            completed_at=row[10], error=row[11],
            result=json.loads(row[12]) if row[12] else {},
        )
        return job

    def update_status(self, job_id: str, status: str, error: str = None):
        job = self.get(job_id)
        if not job: return
        job.status = status
        if status == JobStatus.RUNNING:
            job.started_at = datetime.now().isoformat()
        elif status in (JobStatus.COMPLETE, JobStatus.FAILED):
            job.completed_at = datetime.now().isoformat()
        if error:
            job.error = error
        self.save(job)

    def update_result(self, job_id: str, result: dict):
        job = self.get(job_id)
        if not job: return
        job.result = result
        self.save(job)
