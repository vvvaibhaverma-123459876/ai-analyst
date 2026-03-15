"""
security/security_shell.py
Single entry point. All data passes through this before touching any agent.
All LLM calls are routed through this to enforce masking + policy checks.
"""

from __future__ import annotations
import hashlib
import pandas as pd

from security.data_classifier import DataClassifier
from security.pii_masker import PIIMasker
from security.audit_logger import AuditLogger
from security.policy_store import PolicyStore, PolicyViolation
from security.redaction_engine import RedactionEngine
from security.output_classifier import OutputClassifier
from security.access_controller import AccessController
from security.tenant_isolation import TenantIsolation
from core.logger import get_logger

logger = get_logger(__name__)


class SecurityShell:

    def __init__(
        self,
        tenant_id: str = "default",
        user_id: str = "system",
        policy_path: str = None,
        role: str | None = None,
    ):
        self._tenant = tenant_id
        self._user = user_id
        self._role = (role or '').lower()
        self._classifier = DataClassifier()
        self._masker = PIIMasker()
        self._audit = AuditLogger()
        self._policy = PolicyStore(policy_path)
        self._redactor = RedactionEngine(self._masker)
        self._output_classifier = OutputClassifier()
        self._access = AccessController()
        self._tenant_isolation = TenantIsolation()

    def process_dataframe(self, df: pd.DataFrame, run_id: str = None) -> tuple[pd.DataFrame, dict]:
        classifications = self._classifier.classify_dataframe(df)
        summary = self._classifier.summary(classifications)

        masked_df, mask_report = self._masker.mask_dataframe(df, classifications)

        self._audit.log_ingestion(
            filename="dataframe",
            file_type="structured",
            rows=len(df),
            classification_summary=summary,
            warnings=[],
            run_id=run_id,
            tenant_id=self._tenant,
        )
        if mask_report["masked_columns"] or mask_report["dropped_columns"]:
            self._audit.log_pii_masking(
                mask_report["masked_columns"],
                mask_report["dropped_columns"],
                run_id=run_id,
                tenant_id=self._tenant,
            )

        return masked_df, {
            "classifications": classifications,
            "summary": summary,
            "mask_report": mask_report,
        }

    def llm_complete(
        self,
        system: str,
        user: str,
        run_id: str = None,
        model: str = None,
    ) -> str:
        violation = self._policy.check_internet_off()
        if violation and not self._policy.is_local_llm_mode():
            self._audit.log_policy_block(
                violation.rule, violation.reason, "llm_complete",
                run_id=run_id, tenant_id=self._tenant,
            )
            raise PermissionError(f"Policy block: {violation.reason}")

        masked_system = self._redactor.redact(system)
        masked_user = self._redactor.redact(user)

        if self._policy.is_local_llm_mode():
            cfg = self._policy.local_llm_config()
            from security.local_llm_client import LocalLLMClient
            client = LocalLLMClient(model=cfg["model"], base_url=cfg["base_url"])
            provider = "local_ollama"
            used_model = cfg["model"]
        else:
            from llm.client import LLMClient
            from core.config import config
            client = LLMClient(model=model or config.LLM_MODEL)
            provider = config.LLM_PROVIDER
            used_model = model or config.LLM_MODEL

        self._audit.log_llm_call(
            model=used_model,
            prompt=masked_user,
            provider=provider,
            run_id=run_id,
            tenant_id=self._tenant,
            user_id=self._user,
        )

        return client.complete(masked_system, masked_user)

    def check_finding_policy(self, finding_type: str, metadata: dict, run_id: str = None) -> list[PolicyViolation]:
        violations = []
        n = metadata.get("n", metadata.get("sample_size", None))
        if n is not None:
            v = self._policy.check_sample_size(n)
            if v:
                violations.append(v)
                self._audit.log_policy_block(v.rule, v.reason, finding_type, run_id=run_id, tenant_id=self._tenant)
        if finding_type == "experiment":
            n_a = metadata.get("n_a", 0)
            n_b = metadata.get("n_b", 0)
            v = self._policy.check_ab_sample(n_a, n_b)
            if v:
                violations.append(v)
            p = metadata.get("p_value", 1.0)
            v = self._policy.check_pvalue(p)
            if v:
                violations.append(v)
        return violations

    def check_external_call(self, data_classification: str, run_id: str = None) -> bool:
        v = self._policy.check_internet_off()
        if v:
            self._audit.log_policy_block(v.rule, v.reason, "external_call", run_id=run_id, tenant_id=self._tenant)
            return False
        v = self._policy.check_external_classification(data_classification)
        if v:
            self._audit.log_policy_block(v.rule, v.reason, "external_call", run_id=run_id, tenant_id=self._tenant)
            return False
        return True

    @property
    def audit(self) -> AuditLogger:
        return self._audit

    @property
    def policy(self) -> PolicyStore:
        return self._policy

    @staticmethod
    def data_signature(df: pd.DataFrame) -> str:
        sig = f"{df.shape}|{sorted(df.columns.tolist())}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]

    def classify_output(self, payload) -> str:
        return self._output_classifier.classify(payload)

    def publish_output(self, payload, run_id: str = None, requested_tenant_id: str | None = None):
        requested_tenant_id = requested_tenant_id or self._tenant
        self.assert_access(requested_tenant_id)
        self._tenant_isolation.check_payload(payload, requested_tenant_id)
        classification = self.classify_output(payload)
        if classification in {"CONFIDENTIAL", "RESTRICTED"}:
            payload = self._redact_payload(payload)
        if run_id:
            try:
                self._audit.log_user_action(
                    "OUTPUT_PUBLISH",
                    run_id,
                    detail={"classification": classification},
                    user_id=self._user,
                    tenant_id=self._tenant,
                )
            except Exception:
                pass
        return payload, classification

    def _redact_payload(self, payload):
        if isinstance(payload, str):
            return self._redactor.redact(payload)
        if isinstance(payload, dict):
            return {k: self._redact_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self._redact_payload(v) for v in payload]
        return payload

    def assert_access(self, requested_tenant_id: str, resource_owner: str | None = None) -> None:
        self._tenant_isolation.assert_same_tenant(self._tenant, requested_tenant_id) if self._role != 'admin' and self._user != 'system' and self._tenant != requested_tenant_id else None
        if not self._access.can_access(self._user, self._tenant, requested_tenant_id, role=self._role, resource_owner=resource_owner):
            raise PermissionError('access denied for tenant')
