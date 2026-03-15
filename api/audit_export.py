"""
api/audit_export.py  — v0.6
Audit trail export for SOC 2 / ISO 27001 compliance.

Provides:
  - /admin/audit  GET  — paginated NDJSON or CSV export of audit log
  - /admin/audit/summary  GET  — aggregated stats (calls per tenant, per day)
  - AuditExporter class (also usable standalone / scheduled)

All exports are append-only — no delete endpoint exists by design.
"""

from __future__ import annotations
import csv
import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

from core.logger import get_logger

logger = get_logger(__name__)

AUDIT_DB_PATH = Path(__file__).resolve().parent.parent / "memory" / "audit.db"


class AuditExporter:
    """
    Reads the AuditLogger's SQLite store and exports records
    in a variety of formats suitable for compliance handover.
    """

    def __init__(self, db_path: str = None):
        self._db = db_path or str(AUDIT_DB_PATH)

    def _connect(self):
        import sqlite3
        return sqlite3.connect(self._db)

    # ------------------------------------------------------------------
    # Low-level iterator — all formats build on this
    # ------------------------------------------------------------------

    def iter_records(
        self,
        tenant_id: str = None,
        since: str = None,
        until: str = None,
        limit: int = 10_000,
        offset: int = 0,
    ) -> Iterator[dict]:
        """
        Yields audit records as dicts, newest-first.
        Filters: tenant_id, date range (ISO strings), pagination.
        """
        try:
            import sqlite3
            with self._connect() as conn:
                # Probe actual columns (schema may vary by v0.5 vs v0.6)
                cols = [r[1] for r in conn.execute("PRAGMA table_info(audit_log)").fetchall()]
                if not cols:
                    return
                where_parts = []
                params: list = []
                if tenant_id:
                    if "tenant_id" in cols:
                        where_parts.append("tenant_id = ?")
                        params.append(tenant_id)
                if since:
                    where_parts.append("created_at >= ?")
                    params.append(since)
                if until:
                    where_parts.append("created_at <= ?")
                    params.append(until)
                where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
                params += [limit, offset]
                rows = conn.execute(
                    f"SELECT * FROM audit_log {where} "
                    f"ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    params,
                ).fetchall()
                for row in rows:
                    yield dict(zip(cols, row))
        except Exception as e:
            logger.warning("AuditExporter iter_records failed: %s", e)

    # ------------------------------------------------------------------
    # Export formats
    # ------------------------------------------------------------------

    def to_ndjson(self, **kwargs) -> str:
        """Newline-delimited JSON — each line is a valid JSON object."""
        lines = [json.dumps(r, default=str) for r in self.iter_records(**kwargs)]
        return "\n".join(lines)

    def to_csv(self, **kwargs) -> str:
        """CSV export with header row."""
        records = list(self.iter_records(**kwargs))
        if not records:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
        return output.getvalue()

    def to_json_list(self, **kwargs) -> list[dict]:
        return list(self.iter_records(**kwargs))

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def summary(
        self,
        tenant_id: str = None,
        days: int = 30,
    ) -> dict:
        """
        Returns aggregated stats:
          - total_calls
          - calls_per_day (last N days)
          - calls_per_tenant
          - top_run_ids (most active)
        """
        since = (datetime.now() - timedelta(days=days)).isoformat()
        records = list(self.iter_records(tenant_id=tenant_id, since=since, limit=100_000))

        calls_per_day: dict[str, int] = {}
        calls_per_tenant: dict[str, int] = {}
        calls_per_run: dict[str, int] = {}

        for r in records:
            day = str(r.get("created_at", ""))[:10]
            calls_per_day[day] = calls_per_day.get(day, 0) + 1
            tid = r.get("tenant_id", "default")
            calls_per_tenant[tid] = calls_per_tenant.get(tid, 0) + 1
            run = r.get("run_id", "")
            if run:
                calls_per_run[run] = calls_per_run.get(run, 0) + 1

        top_runs = sorted(calls_per_run.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_records": len(records),
            "period_days": days,
            "generated_at": datetime.now().isoformat(),
            "calls_per_day": calls_per_day,
            "calls_per_tenant": calls_per_tenant,
            "top_run_ids": [{"run_id": r, "calls": c} for r, c in top_runs],
        }


# ------------------------------------------------------------------
# FastAPI router (mounted in api/main.py)
# ------------------------------------------------------------------

def build_audit_router():
    """
    Returns a FastAPI APIRouter with audit export endpoints.
    Mount with: app.include_router(build_audit_router(), prefix="/admin")
    """
    try:
        from fastapi import APIRouter, Depends, Query
        from fastapi.responses import PlainTextResponse, JSONResponse
    except ImportError:
        return None

    router = APIRouter(tags=["audit"])
    exporter = AuditExporter()

    @router.get("/audit", response_class=PlainTextResponse)
    def export_audit(
        format: str = Query("ndjson", description="ndjson | csv"),
        tenant_id: str = Query(None),
        since: str = Query(None, description="ISO datetime"),
        until: str = Query(None, description="ISO datetime"),
        limit: int = Query(5000),
        offset: int = Query(0),
    ):
        """
        Export audit log. Requires admin role.
        Returns NDJSON (default) or CSV.
        """
        kwargs = dict(
            tenant_id=tenant_id, since=since, until=until,
            limit=limit, offset=offset,
        )
        if format == "csv":
            content = exporter.to_csv(**kwargs)
            return PlainTextResponse(content, media_type="text/csv")
        return PlainTextResponse(exporter.to_ndjson(**kwargs), media_type="application/x-ndjson")

    @router.get("/audit/summary")
    def audit_summary(
        tenant_id: str = Query(None),
        days: int = Query(30),
    ):
        """Aggregated audit statistics."""
        return JSONResponse(exporter.summary(tenant_id=tenant_id, days=days))

    return router
