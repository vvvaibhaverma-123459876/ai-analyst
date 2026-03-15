"""
api/main.py
FastAPI service — production API layer.

Endpoints:
  POST   /auth/token          → JWT token
  POST   /jobs                → submit analysis job
  GET    /jobs/{id}           → job status + results
  GET    /jobs/{id}/brief     → final brief only
  GET    /jobs/{id}/findings  → all findings
  POST   /jobs/{id}/verify    → record ground truth outcome
  GET    /memory/context      → org memory for tenant
  POST   /memory/context      → update org context
  GET    /memory/kpis         → KPI definitions
  POST   /memory/kpis         → add KPI definition
  GET    /audit               → audit log (admin only)
  GET    /policy              → current policy
  POST   /policy              → update policy (admin only)
  GET    /health              → health check

Run: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uuid
import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional
import threading

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, UploadFile, File, Form
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.auth import create_token, verify_token, TokenData
from api.job_store import JobStore, Job, JobStatus
from api.rbac import require_role, Role
from security.security_shell import SecurityShell
from security.audit_logger import AuditLogger
from security.policy_store import PolicyStore
from context_engine.org_memory import OrgMemory
from core.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------
app = FastAPI(
    title="AI Analyst API",
    description="Autonomous analytics platform — enterprise API",
    version="0.7.1",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# v0.6: mount audit export router
try:
    from api.audit_export import build_audit_router
    _audit_router = build_audit_router()
    if _audit_router:
        app.include_router(_audit_router, prefix="/admin")
except Exception as _e:
    pass

# v0.6: start scheduler if configured
_scheduler = None
try:
    import os as _os2
    if _os2.getenv("ENABLE_SCHEDULER", "false").lower() == "true":
        from scheduler.monitor import AnalyticsScheduler
        _scheduler = AnalyticsScheduler()
        _scheduler.start()
except Exception as _e2:
    pass

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
job_store = JobStore()
audit = AuditLogger()

# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Issue JWT. In production, verify against your user store."""
    # Minimal demo auth — replace with real user store
    demo_users = {
        os.getenv("ADMIN_USER", "admin"): {
            "password": os.getenv("ADMIN_PASS", "changeme"),
            "role": "admin",
        },
        os.getenv("ANALYST_USER", "analyst"): {
            "password": os.getenv("ANALYST_PASS", "changeme"),
            "role": "analyst",
        },
    }
    user = demo_users.get(form_data.username)
    if not user or user["password"] != form_data.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    token = create_token(
        data={"sub": form_data.username, "role": user["role"]},
        expires_delta=timedelta(hours=8),
    )
    return {"access_token": token, "token_type": "bearer", "expires_in": 28800}


async def current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    data = verify_token(token)
    if not data:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return data


def _assert_job_access(job, user: TokenData) -> None:
    shell = SecurityShell(tenant_id=job.tenant_id, user_id=user.username, role=getattr(user, 'role', None))
    try:
        shell.assert_access(job.tenant_id, resource_owner=job.user_id)
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


# ------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------

class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    error: Optional[str]


@app.post("/jobs", response_model=JobSubmitResponse, tags=["jobs"])
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    kpi_col: Optional[str] = Form(None),
    date_col: Optional[str] = Form(None),
    grain: str = Form("Daily"),
    tenant_id: str = Form("default"),
    user: TokenData = Depends(current_user),
):
    """Submit an analysis job. Returns job_id immediately."""
    require_role(user, [Role.ANALYST, Role.ADMIN])
    SecurityShell(tenant_id=tenant_id, user_id=user.username, role=user.role).assert_access(tenant_id, resource_owner=user.username)

    job_id = str(uuid.uuid4())
    content = await file.read()

    job = Job(
        job_id=job_id,
        tenant_id=tenant_id,
        user_id=user.username,
        filename=file.filename,
        file_content=content,
        kpi_col=kpi_col,
        date_col=date_col,
        grain=grain,
        status=JobStatus.QUEUED,
        created_at=datetime.now().isoformat(),
    )
    job_store.save(job)

    audit.log_user_action(
        "JOB_SUBMIT", job_id,
        detail={"filename": file.filename, "tenant": tenant_id},
        user_id=user.username, tenant_id=tenant_id,
    )

    background_tasks.add_task(_run_job, job_id, tenant_id, user.username)

    return {"job_id": job_id, "status": "queued", "message": "Job queued successfully."}


@app.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str, user: TokenData = Depends(current_user)):
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    _assert_job_access(job, user)
    _assert_job_access(job, user)
    _assert_job_access(job, user)
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error=job.error,
    )


@app.get("/jobs/{job_id}/brief", tags=["jobs"])
async def get_brief(job_id: str, user: TokenData = Depends(current_user)):
    job = job_store.get(job_id)
    if not job or job.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=404, detail="Job not found or not complete")
    _assert_job_access(job, user)
    shell = SecurityShell(tenant_id=job.tenant_id, user_id=user.username, role=user.role)
    payload, classification = shell.publish_output({"job_id": job_id, "brief": job.result.get("brief", ""), "status": job.status}, run_id=job_id, requested_tenant_id=job.tenant_id)
    payload["output_classification"] = classification
    return payload


@app.get("/jobs/{job_id}/findings", tags=["jobs"])
async def get_findings(job_id: str, user: TokenData = Depends(current_user)):
    job = job_store.get(job_id)
    if not job or job.status != JobStatus.COMPLETE:
        raise HTTPException(status_code=404, detail="Job not found or not complete")
    _assert_job_access(job, user)
    shell = SecurityShell(tenant_id=job.tenant_id, user_id=user.username, role=user.role)
    payload, classification = shell.publish_output({"job_id": job_id, "findings": job.result.get("findings", [])}, run_id=job_id, requested_tenant_id=job.tenant_id)
    payload["output_classification"] = classification
    return payload


class VerifyRequest(BaseModel):
    finding_id: str
    outcome_correct: bool
    outcome_actual: str = ""


@app.post("/jobs/{job_id}/verify", tags=["jobs"])
async def verify_finding(
    job_id: str,
    body: VerifyRequest,
    user: TokenData = Depends(current_user),
):
    """Record ground truth outcome for a finding."""
    from ground_truth.recorder import GroundTruthRecorder, Outcome
    recorder = GroundTruthRecorder()
    outcome = Outcome(
        finding_id=body.finding_id,
        outcome_actual=body.outcome_actual,
        outcome_correct=body.outcome_correct,
        verified_by=user.username,
    )
    recorder.record_outcome(outcome)
    audit.log_user_action(
        "VERIFY", body.finding_id,
        detail={"correct": body.outcome_correct},
        user_id=user.username,
    )
    return {"status": "recorded", "finding_id": body.finding_id}


# ------------------------------------------------------------------
# Memory
# ------------------------------------------------------------------

@app.get("/memory/context", tags=["memory"])
async def get_context(tenant_id: str = "default", user: TokenData = Depends(current_user)):
    SecurityShell(tenant_id=tenant_id, user_id=user.username, role=user.role).assert_access(tenant_id, resource_owner=user.username)
    mem = OrgMemory()
    return mem.get_all_context()


@app.post("/memory/context", tags=["memory"])
async def set_context(
    body: dict,
    tenant_id: str = "default",
    user: TokenData = Depends(current_user),
):
    require_role(user, [Role.ANALYST, Role.ADMIN])
    SecurityShell(tenant_id=tenant_id, user_id=user.username, role=user.role).assert_access(tenant_id, resource_owner=user.username)
    mem = OrgMemory()
    for k, v in body.items():
        mem.set(k, v)
    return {"status": "updated", "keys": list(body.keys())}


@app.get("/memory/kpis", tags=["memory"])
async def get_kpis(user: TokenData = Depends(current_user)):
    mem = OrgMemory()
    return mem.all_kpis()


# ------------------------------------------------------------------
# Audit + Policy
# ------------------------------------------------------------------

@app.get("/audit", tags=["admin"])
async def get_audit(n: int = 50, user: TokenData = Depends(current_user)):
    require_role(user, [Role.ADMIN])
    return audit.recent(n=n)


@app.get("/policy", tags=["admin"])
async def get_policy(user: TokenData = Depends(current_user)):
    require_role(user, [Role.ADMIN])
    policy = PolicyStore()
    return policy.all_checks()


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok", "version": "0.7.1", "timestamp": datetime.now().isoformat()}


# ------------------------------------------------------------------
# Background job runner
# ------------------------------------------------------------------

def _run_job(job_id: str, tenant_id: str, user_id: str):
    import io
    import pandas as pd
    from agents.context import AnalysisContext
    from agents.runner import AgentRunner
    from security.security_shell import SecurityShell

    job = job_store.get(job_id)
    if not job:
        return

    job_store.update_status(job_id, JobStatus.RUNNING)

    try:
        shell = SecurityShell(tenant_id=tenant_id, user_id=user_id, role='analyst')
        df = pd.read_csv(io.BytesIO(job.file_content))
        safe_df, sec_report = shell.process_dataframe(df, run_id=job_id)

        context = AnalysisContext(
            df=safe_df,
            date_col=job.date_col or "",
            kpi_col=job.kpi_col or "",
            grain=job.grain or "Daily",
            filename=job.filename,
            security_shell=shell,
            tenant_id=tenant_id,
            user_id=user_id,
            run_id=job_id,
        )

        runner = AgentRunner()
        finished = runner.run(context)

        safe_payload, classification = shell.publish_output({
            "brief": finished.final_brief,
            "follow_up_questions": finished.follow_up_questions,
            "findings": [
                {"agent": name, "summary": r.summary, "status": r.status}
                for name, r in finished.results.items()
            ],
            "agents_run": finished.active_agents,
            "security_report": sec_report.get("summary", ""),
        }, run_id=job_id, requested_tenant_id=tenant_id)
        safe_payload["output_classification"] = classification
        job_store.update_result(job_id, safe_payload)
        job_store.update_status(job_id, JobStatus.COMPLETE)
        logger.info(f"Job {job_id} completed.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job_store.update_status(job_id, JobStatus.FAILED, error=str(e))
