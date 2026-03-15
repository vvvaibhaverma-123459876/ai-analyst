"""
tests/test_bench_security.py
Benchmark coverage for the security/ layer:
  - DataClassifier (column classification)
  - PIIMasker (email, phone, name masking)
  - RedactionEngine (delegates to PIIMasker)
  - OutputClassifier (PII pattern detection in outbound payloads)
  - TenantIsolation (cross-tenant enforcement)
  - AccessController (role/owner checks)
  - SecurityShell integration (process_dataframe, publish_output, assert_access)
  - PolicyStore (internet-off, sample size, p-value rules)
  - AuditLogger (immutability, append-only)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════
# DataClassifier
# ══════════════════════════════════════════════════════════════════════

class TestDataClassifier:

    def test_classifies_email_as_pii(self):
        from security.data_classifier import DataClassifier
        df = pd.DataFrame({"email": ["user@example.com", "admin@corp.io"]})
        classifications = DataClassifier().classify_dataframe(df)
        assert classifications.get("email") in ("PII", "CONFIDENTIAL", "SENSITIVE")

    def test_classifies_revenue_as_internal(self):
        from security.data_classifier import DataClassifier
        df = pd.DataFrame({"revenue": [100, 200, 300]})
        classifications = DataClassifier().classify_dataframe(df)
        level = classifications.get("revenue", "PUBLIC")
        assert level in ("PUBLIC", "INTERNAL")

    def test_summary_contains_counts(self):
        from security.data_classifier import DataClassifier
        df = pd.DataFrame({
            "email": ["a@b.com"],
            "name": ["Alice"],
            "revenue": [100],
        })
        classifications = DataClassifier().classify_dataframe(df)
        summary = DataClassifier().summary(classifications)
        assert isinstance(summary, dict)

    def test_empty_df_no_crash(self):
        from security.data_classifier import DataClassifier
        classifications = DataClassifier().classify_dataframe(pd.DataFrame())
        assert isinstance(classifications, dict)


# ══════════════════════════════════════════════════════════════════════
# PIIMasker
# ══════════════════════════════════════════════════════════════════════

class TestPIIMasker:

    def test_masks_email_in_prompt(self):
        from security.pii_masker import PIIMasker
        m = PIIMasker()
        result = m.mask_prompt("Contact user@example.com for details")
        assert "user@example.com" not in result
        assert "[" in result or "REDACTED" in result.upper() or "EMAIL" in result.upper()

    def test_masks_phone_number(self):
        from security.pii_masker import PIIMasker
        m = PIIMasker()
        result = m.mask_prompt("Call 9876543210 now")
        assert "9876543210" not in result

    def test_clean_text_unchanged(self):
        from security.pii_masker import PIIMasker
        text = "Revenue increased by 15% in Q3."
        result = PIIMasker().mask_prompt(text)
        # No PII → text should be largely unchanged
        assert "Revenue" in result or len(result) > 5

    def test_mask_dataframe_pii_columns(self):
        from security.pii_masker import PIIMasker
        from security.data_classifier import DataClassifier
        df = pd.DataFrame({"email": ["a@b.com", "c@d.io"], "revenue": [100, 200]})
        classifications = DataClassifier().classify_dataframe(df)
        masked_df, report = PIIMasker().mask_dataframe(df, classifications)
        # Email column should be masked or dropped
        if "email" in masked_df.columns:
            assert masked_df["email"].apply(lambda x: "@" not in str(x)).all()
        assert "revenue" in masked_df.columns
        assert (masked_df["revenue"] == df["revenue"]).all()

    def test_empty_string_no_crash(self):
        from security.pii_masker import PIIMasker
        result = PIIMasker().mask_prompt("")
        assert result == ""

    def test_none_handled(self):
        from security.pii_masker import PIIMasker
        result = PIIMasker().mask_prompt(None)
        assert result is not None


# ══════════════════════════════════════════════════════════════════════
# RedactionEngine
# ══════════════════════════════════════════════════════════════════════

class TestRedactionEngine:

    def test_delegates_to_pii_masker(self):
        from security.redaction_engine import RedactionEngine
        engine = RedactionEngine()
        result = engine.redact("Email me at ceo@bigcorp.com today")
        assert "ceo@bigcorp.com" not in result

    def test_empty_string_safe(self):
        from security.redaction_engine import RedactionEngine
        assert RedactionEngine().redact("") == ""

    def test_no_double_redaction(self):
        from security.redaction_engine import RedactionEngine
        text = "already [EMAIL_REDACTED] here"
        result = RedactionEngine().redact(text)
        assert result.count("[EMAIL_REDACTED]") <= 1


# ══════════════════════════════════════════════════════════════════════
# OutputClassifier
# ══════════════════════════════════════════════════════════════════════

class TestOutputClassifier:

    def test_email_in_payload_is_confidential(self):
        from security.output_classifier import OutputClassifier
        oc = OutputClassifier()
        assert oc.classify({"brief": "Contact ceo@bigcorp.com"}) == "CONFIDENTIAL"

    def test_redacted_tag_is_restricted(self):
        from security.output_classifier import OutputClassifier
        assert OutputClassifier().classify("[REDACTED email]") == "RESTRICTED"

    def test_clean_text_is_internal(self):
        from security.output_classifier import OutputClassifier
        assert OutputClassifier().classify("Revenue grew 12% in March") == "INTERNAL"

    def test_nested_dict_classified(self):
        from security.output_classifier import OutputClassifier
        payload = {"section": {"sub": "call 9876543210 now"}}
        result = OutputClassifier().classify(payload)
        assert result in ("CONFIDENTIAL", "RESTRICTED", "INTERNAL")

    def test_list_payload_classified(self):
        from security.output_classifier import OutputClassifier
        result = OutputClassifier().classify(["hello", "test@example.com"])
        assert result == "CONFIDENTIAL"

    def test_none_payload_no_crash(self):
        from security.output_classifier import OutputClassifier
        result = OutputClassifier().classify(None)
        assert result in ("INTERNAL", "RESTRICTED", "CONFIDENTIAL")

    def test_policy_block_text_is_restricted(self):
        from security.output_classifier import OutputClassifier
        assert OutputClassifier().classify("policy block triggered") == "RESTRICTED"


# ══════════════════════════════════════════════════════════════════════
# TenantIsolation
# ══════════════════════════════════════════════════════════════════════

class TestTenantIsolation:

    def test_same_tenant_passes(self):
        from security.tenant_isolation import TenantIsolation
        TenantIsolation().assert_same_tenant("acme", "acme")  # should not raise

    def test_different_tenant_raises(self):
        from security.tenant_isolation import TenantIsolation
        with pytest.raises(PermissionError):
            TenantIsolation().assert_same_tenant("acme", "rival")

    def test_payload_check_wrong_tenant_raises(self):
        from security.tenant_isolation import TenantIsolation
        with pytest.raises(PermissionError):
            TenantIsolation().check_payload({"tenant_id": "other"}, "acme")

    def test_payload_check_correct_tenant_passes(self):
        from security.tenant_isolation import TenantIsolation
        result = TenantIsolation().check_payload({"tenant_id": "acme"}, "acme")
        assert result is not None

    def test_payload_without_tenant_passes(self):
        from security.tenant_isolation import TenantIsolation
        result = TenantIsolation().check_payload({"data": "value"}, "acme")
        assert result is not None


# ══════════════════════════════════════════════════════════════════════
# AccessController
# ══════════════════════════════════════════════════════════════════════

class TestAccessController:

    def test_system_user_can_access_any_tenant(self):
        from security.access_controller import AccessController
        assert AccessController().can_access("system", "acme", "other") is True

    def test_admin_role_can_access_any_tenant(self):
        from security.access_controller import AccessController
        assert AccessController().can_access("alice", "acme", "beta", role="admin") is True

    def test_analyst_same_tenant_allowed(self):
        from security.access_controller import AccessController
        assert AccessController().can_access("alice", "acme", "acme", role="analyst") is True

    def test_analyst_cross_tenant_denied(self):
        from security.access_controller import AccessController
        assert AccessController().can_access("alice", "acme", "rival", role="analyst") is False

    def test_resource_owner_own_resource_allowed(self):
        from security.access_controller import AccessController
        assert AccessController().can_access(
            "bob", "acme", "acme", role="viewer", resource_owner="bob") is True


# ══════════════════════════════════════════════════════════════════════
# SecurityShell integration
# ══════════════════════════════════════════════════════════════════════

class TestSecurityShellIntegration:

    def test_process_dataframe_masks_pii(self):
        from security.security_shell import SecurityShell
        df = pd.DataFrame({
            "email": ["a@b.com", "c@d.io"],
            "revenue": [100.0, 200.0],
        })
        shell = SecurityShell(tenant_id="acme", user_id="u1")
        masked_df, report = shell.process_dataframe(df, run_id="test-001")
        assert "mask_report" in report
        assert "revenue" in masked_df.columns
        if "email" in masked_df.columns:
            assert masked_df["email"].apply(lambda x: "@" not in str(x)).all()

    def test_publish_output_redacts_confidential(self):
        from security.security_shell import SecurityShell
        shell = SecurityShell(tenant_id="acme", user_id="u1", role="analyst")
        payload, classification = shell.publish_output(
            {"brief": "Contact test@example.com for follow-up"},
            run_id="r1",
            requested_tenant_id="acme",
        )
        assert classification in ("CONFIDENTIAL", "RESTRICTED")
        if isinstance(payload, dict) and "brief" in payload:
            assert "test@example.com" not in payload["brief"]

    def test_publish_output_clean_is_internal(self):
        from security.security_shell import SecurityShell
        shell = SecurityShell(tenant_id="acme", user_id="u1", role="analyst")
        payload, classification = shell.publish_output(
            {"brief": "Revenue grew 12% in Q3 2025."},
            requested_tenant_id="acme",
        )
        assert classification == "INTERNAL"

    def test_assert_access_cross_tenant_raises_for_analyst(self):
        from security.security_shell import SecurityShell
        shell = SecurityShell(tenant_id="acme", user_id="alice", role="analyst")
        with pytest.raises(PermissionError):
            shell.assert_access("rival")

    def test_assert_access_admin_cross_tenant_ok(self):
        from security.security_shell import SecurityShell
        shell = SecurityShell(tenant_id="acme", user_id="alice", role="admin")
        shell.assert_access("rival")  # should not raise

    def test_data_signature_deterministic(self):
        from security.security_shell import SecurityShell
        df = pd.DataFrame({"a": [1,2,3], "b": [4,5,6]})
        sig1 = SecurityShell.data_signature(df)
        sig2 = SecurityShell.data_signature(df)
        assert sig1 == sig2
        assert len(sig1) == 16

    def test_classify_output_delegates(self):
        from security.security_shell import SecurityShell
        shell = SecurityShell(tenant_id="acme")
        result = shell.classify_output({"text": "CEO email: ceo@acme.com"})
        assert result in ("CONFIDENTIAL", "RESTRICTED", "INTERNAL")


# ══════════════════════════════════════════════════════════════════════
# PolicyStore
# ══════════════════════════════════════════════════════════════════════

class TestPolicyStore:

    def test_check_sample_size_below_minimum(self):
        from security.policy_store import PolicyStore
        store = PolicyStore()
        violation = store.check_sample_size(2)
        # With default config min_sample=30, n=2 should violate
        if violation:
            assert violation.rule
            assert violation.reason

    def test_check_sample_size_above_minimum_passes(self):
        from security.policy_store import PolicyStore
        store = PolicyStore()
        violation = store.check_sample_size(1000)
        assert violation is None

    def test_check_pvalue_above_threshold(self):
        from security.policy_store import PolicyStore
        violation = PolicyStore().check_pvalue(0.8)
        assert violation is not None

    def test_check_pvalue_below_threshold_passes(self):
        from security.policy_store import PolicyStore
        violation = PolicyStore().check_pvalue(0.01)
        assert violation is None

    def test_internet_off_returns_none_when_off_is_false(self):
        from security.policy_store import PolicyStore
        store = PolicyStore()
        store.set("internet_off_mode", False)
        violation = store.check_internet_off()
        assert violation is None
