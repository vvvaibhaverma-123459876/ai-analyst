"""
security/policy_store.py
Admin-defined policy rules stored in YAML.
These rules are NOT learnable — they can only be changed by an admin
with the change logged to the audit trail.

Policy rules govern:
  - minimum sample sizes before a finding can be published
  - required confidence intervals on forecasts
  - p-value thresholds for experiment conclusions
  - prohibition on causal claims without a DAG
  - maximum data classification level allowed in external calls
  - internet-off mode (zero external calls)
  - local LLM mode
  - which agents are permitted to run
"""

from __future__ import annotations
import yaml
import os
from pathlib import Path
from dataclasses import dataclass, field
from core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_POLICY_PATH = Path(__file__).resolve().parent.parent / "configs" / "policy.yaml"

DEFAULT_POLICY = {
    "min_sample_size": 30,
    "min_ab_group_size": 100,
    "max_pvalue_threshold": 0.05,
    "require_confidence_intervals": True,
    "no_causal_claims_without_dag": True,
    "min_forecast_data_points": 14,
    "min_cluster_silhouette": 0.25,
    "max_anomaly_false_positive_rate": 0.20,
    "max_external_data_classification": "PUBLIC",
    "internet_off_mode": False,
    "local_llm_mode": False,
    "local_llm_model": "mistral",
    "local_llm_base_url": "http://localhost:11434",
    "permitted_agents": [
        "eda", "orchestrator", "trend", "anomaly", "root_cause",
        "funnel", "cohort", "forecast", "experiment", "ml_cluster",
        "nlp", "vision", "debate", "insight",
    ],
    "require_debate_jury": True,
    "max_confidence_without_holdout": 0.70,
    "allow_raw_data_in_prompts": False,
}


@dataclass
class PolicyViolation:
    rule: str
    reason: str
    value_found: object
    value_required: object
    blocking: bool = True


class PolicyStore:

    def __init__(self, policy_path: str = None):
        self._path = policy_path or str(DEFAULT_POLICY_PATH)
        self._policy = self._load()

    def _load(self) -> dict:
        if os.path.exists(self._path):
            try:
                with open(self._path, "r") as f:
                    loaded = yaml.safe_load(f) or {}
                merged = {**DEFAULT_POLICY, **loaded}
                logger.info(f"Policy loaded from {self._path}")
                return merged
            except Exception as e:
                logger.warning(f"Policy load failed: {e}. Using defaults.")
        return DEFAULT_POLICY.copy()

    def get(self, key: str, default=None):
        return self._policy.get(key, default)

    # ------------------------------------------------------------------
    # Check methods — called by Guardian before publishing findings
    # ------------------------------------------------------------------

    def check_sample_size(self, n: int) -> PolicyViolation | None:
        min_n = self._policy["min_sample_size"]
        if n < min_n:
            return PolicyViolation(
                rule="min_sample_size",
                reason=f"Sample size {n} < required {min_n}",
                value_found=n, value_required=min_n,
            )
        return None

    def check_ab_sample(self, n_a: int, n_b: int) -> PolicyViolation | None:
        min_n = self._policy["min_ab_group_size"]
        smallest = min(n_a, n_b)
        if smallest < min_n:
            return PolicyViolation(
                rule="min_ab_group_size",
                reason=f"Smallest A/B group {smallest} < required {min_n}",
                value_found=smallest, value_required=min_n,
            )
        return None

    def check_pvalue(self, p: float) -> PolicyViolation | None:
        threshold = self._policy["max_pvalue_threshold"]
        if p > threshold:
            return PolicyViolation(
                rule="max_pvalue_threshold",
                reason=f"p-value {p:.4f} exceeds threshold {threshold}",
                value_found=p, value_required=threshold, blocking=False,
            )
        return None

    def check_external_classification(self, level: str) -> PolicyViolation | None:
        allowed = self._policy["max_external_data_classification"]
        order = ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "SENSITIVE", "PII"]
        if order.index(level) > order.index(allowed):
            return PolicyViolation(
                rule="max_external_data_classification",
                reason=f"Data classification '{level}' exceeds allowed '{allowed}' for external calls",
                value_found=level, value_required=allowed,
            )
        return None

    def check_internet_off(self) -> PolicyViolation | None:
        if self._policy["internet_off_mode"]:
            return PolicyViolation(
                rule="internet_off_mode",
                reason="Internet-off mode is active. No external calls permitted.",
                value_found=True, value_required=False,
            )
        return None

    def is_local_llm_mode(self) -> bool:
        return self._policy.get("local_llm_mode", False)

    def local_llm_config(self) -> dict:
        return {
            "model": self._policy.get("local_llm_model", "mistral"),
            "base_url": self._policy.get("local_llm_base_url", "http://localhost:11434"),
        }

    def check_agent_permitted(self, agent_name: str) -> PolicyViolation | None:
        permitted = self._policy.get("permitted_agents", [])
        if agent_name not in permitted:
            return PolicyViolation(
                rule="permitted_agents",
                reason=f"Agent '{agent_name}' is not in the permitted agents list",
                value_found=agent_name, value_required=permitted,
            )
        return None

    def all_checks(self) -> dict:
        """Returns full policy as dict for display."""
        return dict(self._policy)
