"""
security/security_shell.py
Single entry point. All data passes through this before touching any agent.
All LLM calls are routed through this to enforce masking + policy checks.

Usage:
    shell = SecurityShell(tenant_id="zet", user_id="baba")
    safe_df, report = shell.process_dataframe(df)
    response = shell.llm_complete(system, user, run_id=run_id)
"""

from __future__ import annotations
import hashlib
import pandas as pd

from security.data_classifier import DataClassifier
from security.pii_masker import PIIMasker
from security.audit_logger import AuditLogger
from security.policy_store import PolicyStore, PolicyViolation
from core.logger import get_logger

logger = get_logger(__name__)


class SecurityShell:

    def __init__(
        self,
        tenant_id: str = "default",
        user_id: str = "system",
        policy_path: str = None,
    ):
        self._tenant = tenant_id
        self._user = user_id
        self._classifier = DataClassifier()
        self._masker = PIIMasker()
        self._audit = AuditLogger()
        self._policy = PolicyStore(policy_path)

    # ------------------------------------------------------------------
    # DataFrame processing
    # ------------------------------------------------------------------

    def process_dataframe(self, df: pd.DataFrame, run_id: str = None) -> tuple[pd.DataFrame, dict]:
        """
        Classify → mask PII → log → return safe DataFrame.
        """
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

    # ------------------------------------------------------------------
    # LLM routing — enforces policy, masking, local mode
    # ------------------------------------------------------------------

    def llm_complete(
        self,
        system: str,
        user: str,
        run_id: str = None,
        model: str = None,
    ) -> str:
        """
        Route an LLM completion through the security shell.
        Enforces: internet-off mode, local LLM mode, prompt masking, audit logging.
        """
        # Check internet-off policy
        violation = self._policy.check_internet_off()
        if violation and not self._policy.is_local_llm_mode():
            self._audit.log_policy_block(
                violation.rule, violation.reason, "llm_complete",
                run_id=run_id, tenant_id=self._tenant,
            )
            raise PermissionError(f"Policy block: {violation.reason}")

        # Mask prompt content
        masked_system = self._masker.mask_prompt(system)
        masked_user = self._masker.mask_prompt(user)

        # Route to local or remote LLM
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

        # Audit the call (hash only, never content)
        self._audit.log_llm_call(
            model=used_model,
            prompt=masked_user,
            provider=provider,
            run_id=run_id,
            tenant_id=self._tenant,
            user_id=self._user,
        )

        return client.complete(masked_system, masked_user)

    # ------------------------------------------------------------------
    # Policy checking — called before publishing findings
    # ------------------------------------------------------------------

    def check_finding_policy(
        self,
        finding_type: str,
        metadata: dict,
        run_id: str = None,
    ) -> list[PolicyViolation]:
        """
        Runs all applicable policy checks for a finding type.
        Returns list of violations (may be empty).
        Blocking violations prevent publication.
        """
        violations = []

        n = metadata.get("n", metadata.get("sample_size", None))
        if n is not None:
            v = self._policy.check_sample_size(n)
            if v:
                violations.append(v)
                self._audit.log_policy_block(
                    v.rule, v.reason, finding_type,
                    run_id=run_id, tenant_id=self._tenant,
                )

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
        """Returns True if external call is permitted."""
        # Internet-off check
        v = self._policy.check_internet_off()
        if v:
            self._audit.log_policy_block(
                v.rule, v.reason, "external_call",
                run_id=run_id, tenant_id=self._tenant,
            )
            return False

        # Classification check
        v = self._policy.check_external_classification(data_classification)
        if v:
            self._audit.log_policy_block(
                v.rule, v.reason, "external_call",
                run_id=run_id, tenant_id=self._tenant,
            )
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
        """Stable hash of data shape + column names — not row content."""
        sig = f"{df.shape}|{sorted(df.columns.tolist())}"
        return hashlib.sha256(sig.encode()).hexdigest()[:16]
