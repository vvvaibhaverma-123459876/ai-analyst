"""
security/data_classifier.py
Classifies every DataFrame column before it touches any LLM.

Classification levels:
  PUBLIC      — safe to send externally (aggregates, non-personal metrics)
  INTERNAL    — company data, not personal, restricted externally
  CONFIDENTIAL — business-sensitive (revenue, pricing, strategies)
  PII         — personally identifiable (name, email, phone, ID)
  SENSITIVE   — health, financial account, legal, biometric

Policy: only PUBLIC columns can leave the security boundary.
All others are masked or excluded before any external API call.
"""

from __future__ import annotations
import re
import pandas as pd
from dataclasses import dataclass
from core.logger import get_logger

logger = get_logger(__name__)

# Column name patterns → classification
_PII_PATTERNS = [
    r"\bname\b", r"\bemail\b", r"\bphone\b", r"\bmobile\b",
    r"\baddress\b", r"\bstreet\b", r"\bzip\b", r"\bpostcode\b",
    r"\bpassport\b", r"\bnational_id\b", r"\bssn\b", r"\bpan\b",
    r"\baadhar\b", r"\buid\b", r"\buser_id\b", r"\bcustomer_id\b",
    r"\bip_address\b", r"\bmac_address\b", r"\bdevice_id\b",
    r"\bdob\b", r"\bbirthdate\b", r"\bbirthday\b",
    r"\bgps\b", r"\blat\b", r"\blon\b", r"\blocation\b",
]

_SENSITIVE_PATTERNS = [
    r"\baccount\b", r"\biban\b", r"\bcredit\b", r"\bcard\b",
    r"\bsalary\b", r"\bwage\b", r"\bincome\b", r"\bpayment\b",
    r"\bhealth\b", r"\bdiagnos\b", r"\bmedical\b", r"\bprescri\b",
    r"\blegal\b", r"\bcontract\b", r"\bpassword\b", r"\btoken\b",
    r"\bsecret\b", r"\bapi_key\b",
]

_CONFIDENTIAL_PATTERNS = [
    r"\brevenue\b", r"\bprofit\b", r"\bcost\b", r"\bprice\b",
    r"\bmargin\b", r"\bforecast\b", r"\bbudget\b", r"\bsales\b",
    r"\bchurn\b", r"\bltv\b", r"\bclv\b",
]

_VALUE_PII_PATTERNS = [
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",     # email
    r"^\+?[\d\s\-\(\)]{8,15}$",                                  # phone
    r"^\d{3}-\d{2}-\d{4}$",                                      # SSN
    r"^[2-9]\d{11}$",                                             # Aadhaar
]


@dataclass
class ColumnClassification:
    column: str
    level: str          # PUBLIC | INTERNAL | CONFIDENTIAL | PII | SENSITIVE
    reason: str
    safe_for_external: bool


class DataClassifier:

    def classify_dataframe(self, df: pd.DataFrame) -> dict[str, ColumnClassification]:
        results = {}
        for col in df.columns:
            cls = self._classify_column(col, df[col])
            results[col] = cls
            if cls.level in ("PII", "SENSITIVE"):
                logger.warning(f"Column '{col}' classified as {cls.level}: {cls.reason}")
        return results

    def _classify_column(self, name: str, series: pd.Series) -> ColumnClassification:
        name_lower = name.lower()

        # Check PII by name
        for pat in _PII_PATTERNS:
            if re.search(pat, name_lower):
                return ColumnClassification(name, "PII", f"name matches {pat}", False)

        # Check SENSITIVE by name
        for pat in _SENSITIVE_PATTERNS:
            if re.search(pat, name_lower):
                return ColumnClassification(name, "SENSITIVE", f"name matches {pat}", False)

        # Check CONFIDENTIAL by name
        for pat in _CONFIDENTIAL_PATTERNS:
            if re.search(pat, name_lower):
                return ColumnClassification(name, "CONFIDENTIAL", f"name matches {pat}", False)

        # Check values for PII patterns (sample first 50 non-null values)
        if series.dtype == object:
            sample = series.dropna().astype(str).head(50)
            for val in sample:
                for pat in _VALUE_PII_PATTERNS:
                    if re.match(pat, val.strip()):
                        return ColumnClassification(
                            name, "PII", f"values match PII pattern {pat}", False
                        )

        # High-cardinality string → likely internal
        if series.dtype == object and series.nunique() > 100:
            return ColumnClassification(name, "INTERNAL", "high-cardinality string", False)

        # Default: PUBLIC
        return ColumnClassification(name, "PUBLIC", "no sensitive signal detected", True)

    def safe_columns(self, classifications: dict[str, ColumnClassification]) -> list[str]:
        return [col for col, cls in classifications.items() if cls.safe_for_external]

    def pii_columns(self, classifications: dict[str, ColumnClassification]) -> list[str]:
        return [col for col, cls in classifications.items() if cls.level == "PII"]

    def summary(self, classifications: dict[str, ColumnClassification]) -> str:
        from collections import Counter
        counts = Counter(cls.level for cls in classifications.values())
        return " | ".join(f"{level}: {n}" for level, n in sorted(counts.items()))
