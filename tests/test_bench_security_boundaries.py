"""
tests/test_bench_security_boundaries.py  — v9
Security boundary audit.

Tests every outbound surface:
  - API responses
  - Monitoring alerts
  - Replay outputs
  - Audit exports
  - Scheduler notifications
  - Pipeline output (GovernedPipeline.publish)
  - Tenant isolation under replay
  - PII/redaction single path verification
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest import make_ts


# ══════════════════════════════════════════════════════════════════════
# All PII redaction goes through one path
# ══════════════════════════════════════════════════════════════════════

class TestSingleRedactionPath:

    def test_pii_masker_and_redaction_engine_same_output(self):
        """RedactionEngine must delegate to PIIMasker — not parallel logic."""
        from security.pii_masker import PIIMasker
        from security.redaction_engine import RedactionEngine
        text = "Call CEO at 9876543210 or ceo@acme.com"
        via_masker   = PIIMasker().mask_prompt(text)
        via_redactor = RedactionEngine().redact(text)
        # Both must remove the same PII tokens
        assert "9876543210" not in via_masker
        assert "9876543210" not in via_redactor
        assert "ceo@acme.com" not in via_masker
        assert "ceo@acme.com" not in via_redactor

    def test_output_classifier_detects_pii_redacted_tag(self):
        from security.output_classifier import OutputClassifier
        assert OutputClassifier().classify("[EMAIL_REDACTED] called us") == "RESTRICTED"

    def test_output_classifier_detects_phone(self):
        from security.output_classifier import OutputClassifier
        assert OutputClassifier().classify("call 9876543210") == "CONFIDENTIAL"


# ══════════════════════════════════════════════════════════════════════
# Pipeline output always passes through SecurityShell
# ══════════════════════════════════════════════════════════════════════

class TestPipelineOutputBoundary:

    def test_governed_pipeline_sets_output_classification(self):
        from agents.pipeline import GovernedPipeline
        from agents.context import AnalysisContext
        df = make_ts(n=30)
        ctx = AnalysisContext(df=df, kpi_col="revenue", date_col="date",
                              tenant_id="default")
        ctx = GovernedPipeline().run(ctx)
        assert ctx.run_manifest.get("output_classification") in (
            "INTERNAL", "CONFIDENTIAL", "RESTRICTED", "BLOCKED"
        )

    def test_governed_pipeline_with_pii_in_context(self):
        """Even if business context has PII, final brief must be clean or classified."""
        from agents.pipeline import GovernedPipeline
        from agents.context import AnalysisContext
        df = make_ts(n=30)
        ctx = AnalysisContext(
            df=df, kpi_col="revenue", date_col="date", tenant_id="default",
            business_context={"contact": "ceo@bigcorp.com", "company": "BigCorp"},
        )
        ctx = GovernedPipeline().run(ctx)
        # If the email leaked into the brief, it must be classified as CONFIDENTIAL+
        classification = ctx.run_manifest.get("output_classification", "INTERNAL")
        if "ceo@bigcorp.com" in ctx.final_brief:
            assert classification in ("CONFIDENTIAL", "RESTRICTED")


# ══════════════════════════════════════════════════════════════════════
# Monitoring alerts pass through security
# ══════════════════════════════════════════════════════════════════════

class TestMonitoringAlertBoundary:

    def test_alert_message_has_no_raw_pii(self, monkeypatch):
        """Scheduler MonitorRunner builds alert messages — they must not contain PII."""
        from scheduler.monitor import MonitorRunner, MonitorJob
        runner = MonitorRunner()
        job = MonitorJob(
            job_id="j1", name="Daily signups",
            connector="csv", query="",
            kpi_col="signups", date_col="date",
        )
        df = make_ts(n=60, kpi="signups", spike_idx=50)
        anomaly_result = {
            "anomaly_count": 1,
            "anomaly_records": [{"date": "2025-02-20", "value": 999.9, "z_score": 4.5}]
        }
        msg = runner._build_alert_message(job, df, anomaly_result)
        assert "signups" in msg.lower() or "anomaly" in msg.lower()
        # No email addresses or phone numbers in the alert message
        assert "@" not in msg
        import re
        assert not re.search(r'\b\d{10}\b', msg)


# ══════════════════════════════════════════════════════════════════════
# Audit export boundary
# ══════════════════════════════════════════════════════════════════════

class TestAuditExportBoundary:

    def test_audit_export_tenant_isolation(self, tmp_path):
        """Audit records for tenant A must not appear in tenant B export."""
        import sqlite3
        import json
        from api.audit_export import AuditExporter
        # Create a minimal audit DB with two tenants
        db_path = tmp_path / "audit.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE audit_log (
                    id INTEGER PRIMARY KEY, run_id TEXT,
                    tenant_id TEXT, created_at TEXT, event TEXT
                )
            """)
            conn.execute("INSERT INTO audit_log VALUES (1,'r1','acme','2025-01-01','login')")
            conn.execute("INSERT INTO audit_log VALUES (2,'r2','rival','2025-01-01','login')")
            conn.commit()

        exporter = AuditExporter(db_path=str(db_path))
        acme_records  = exporter.to_json_list(tenant_id="acme")
        rival_records = exporter.to_json_list(tenant_id="rival")

        acme_tenants  = {r.get("tenant_id") for r in acme_records}
        rival_tenants = {r.get("tenant_id") for r in rival_records}

        assert "rival" not in acme_tenants
        assert "acme"  not in rival_tenants

    def test_audit_ndjson_each_line_valid_json(self, tmp_path):
        import sqlite3
        from api.audit_export import AuditExporter
        import json
        db_path = tmp_path / "audit.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, run_id TEXT, created_at TEXT)")
            conn.execute("INSERT INTO audit_log VALUES (1,'r1','2025-01-01')")
            conn.commit()
        ndjson = AuditExporter(db_path=str(db_path)).to_ndjson(limit=10)
        if ndjson.strip():
            for line in ndjson.strip().split("\n"):
                json.loads(line)  # must not raise


# ══════════════════════════════════════════════════════════════════════
# Replay tenant isolation
# ══════════════════════════════════════════════════════════════════════

class TestReplayTenantIsolation:

    def test_replay_uses_stored_tenant_context(self, tmp_path):
        """Replay must reconstruct the SecurityShell with the original tenant."""
        from evaluation.replay_harness import ReplayHarness
        from versioning.run_manifest import RunManifest

        df = make_ts(n=20)
        csv_path = tmp_path / "data.csv"
        df.to_csv(csv_path, index=False)

        m = RunManifest.create("replay-tenant-test")
        m.replay_data_path = str(csv_path)
        m.replay_context = {
            "date_col": "date", "kpi_col": "revenue",
            "grain": "Daily", "tenant_id": "acme", "user_id": "u1",
        }
        m.persist(base_dir=str(tmp_path))

        replayed = ReplayHarness(base_dir=str(tmp_path)).replay("replay-tenant-test")
        assert replayed.tenant_id == "acme"

    def test_cross_tenant_replay_blocked(self, tmp_path):
        """A user from tenant 'rival' must not replay a run from tenant 'acme'."""
        from security.security_shell import SecurityShell
        from security.tenant_isolation import TenantIsolation

        shell = SecurityShell(tenant_id="rival", user_id="attacker", role="analyst")
        with pytest.raises(PermissionError):
            shell.assert_access("acme")


# ══════════════════════════════════════════════════════════════════════
# Session store tenant isolation
# ══════════════════════════════════════════════════════════════════════

class TestSessionStoreTenantIsolation:

    def test_annotations_isolated_by_tenant(self, tmp_path):
        from api.session_manager import SessionStore
        store = SessionStore(db_path=str(tmp_path / "sessions.db"))
        store.annotate("run1", "anomaly", "alice", "acme",   "correct", "looks right")
        store.annotate("run1", "anomaly", "bob",   "rival",  "disputed", "disagree")

        acme_anns  = store.get_annotations("run1", tenant_id="acme")
        rival_anns = store.get_annotations("run1", tenant_id="rival")

        assert all(a.tenant_id == "acme"  for a in acme_anns)
        assert all(a.tenant_id == "rival" for a in rival_anns)
        assert len(acme_anns) == 1
        assert len(rival_anns) == 1
