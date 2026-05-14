"""
security/data_classifier.py
Conservative, analytics-friendly data classifier.

The classifier intentionally separates raw identifiers/PII from ordinary
business metrics. Revenue, sales, conversion, and counts are INTERNAL by
default for governance, but are not treated as PII or binned/masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import re
import pandas as pd

from core.logger import get_logger

logger = get_logger(__name__)

_PII_NAME_PATTERNS = [
    r"\bname\b", r"\bemail\b", r"\be[-_ ]?mail\b", r"\bphone\b", r"\bmobile\b",
    r"\baddress\b", r"\bstreet\b", r"\bpassport\b", r"\bnational_id\b", r"\bssn\b",
    r"\bpan\b", r"\baadhaar\b", r"\baadhar\b", r"\bip_address\b", r"\bmac_address\b",
    r"\bdevice_id\b", r"\bdob\b", r"\bbirthdate\b", r"\bbirthday\b",
    r"\bgps\b", r"\blat\b", r"\blon\b", r"\blocation\b",
]

# IDs are often join keys needed for analysis. Keep them internal rather than PII
# unless values themselves look like known PII identifiers.
_INTERNAL_NAME_PATTERNS = [
    r"\buid\b", r"\buser_id\b", r"\bcustomer_id\b", r"\border_id\b", r"\bsession_id\b",
    r"\bzip\b", r"\bpostcode\b", r"\bpincode\b",
]

_SENSITIVE_NAME_PATTERNS = [
    r"\biban\b", r"\baccount_number\b", r"\bpassword\b", r"\btoken\b", r"\bsecret\b", r"\bapi_key\b",
    r"\bhealth\b", r"\bdiagnos", r"\bmedical\b", r"\bprescri", r"\blegal\b",
]

_INTERNAL_BUSINESS_PATTERNS = [
    r"\brevenue\b", r"\bprofit\b", r"\bcost\b", r"\bprice\b", r"\bmargin\b",
    r"\bforecast\b", r"\bbudget\b", r"\bsales\b", r"\bchurn\b", r"\bltv\b", r"\bclv\b",
    r"\bincome\b", r"\bsalary\b", r"\bpayment\b", r"\bcredit\b", r"\bcard\b",
]

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
_PAN_RE = re.compile(r"^[A-Z]{5}\d{4}[A-Z]$")
_AADHAAR_RE = re.compile(r"^[2-9]\d{11}$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_PHONE_RE = re.compile(r"^(?:\+?91[-\s]?)?[6-9]\d{9}$")
_PHONE_SEPARATED_RE = re.compile(r"^\+?(?:\d[\s\-()]){7,}\d$")
_DATE_LIKE_RE = re.compile(
    r"^(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{8})$"
)


@dataclass(frozen=True)
class ColumnClassification:
    column: str
    level: str          # PUBLIC | INTERNAL | CONFIDENTIAL | PII | SENSITIVE
    reason: str
    safe_for_external: bool

    def __eq__(self, other):  # backward-compatible with older tests/callers
        if isinstance(other, str):
            return self.level == other
        return super().__eq__(other)

    def __hash__(self):
        return hash((self.column, self.level, self.reason, self.safe_for_external))


class DataClassifier:
    def classify_dataframe(self, df: pd.DataFrame) -> dict[str, ColumnClassification]:
        results: dict[str, ColumnClassification] = {}
        for col in df.columns:
            cls = self._classify_column(col, df[col])
            results[col] = cls
            if cls.level in ("PII", "SENSITIVE"):
                logger.warning("Column '%s' classified as %s: %s", col, cls.level, cls.reason)
        return results

    def _classify_column(self, name: str, series: pd.Series) -> ColumnClassification:
        name_lower = str(name).lower()

        for pat in _PII_NAME_PATTERNS:
            if re.search(pat, name_lower):
                return ColumnClassification(name, "PII", f"name matches {pat}", False)

        for pat in _SENSITIVE_NAME_PATTERNS:
            if re.search(pat, name_lower):
                return ColumnClassification(name, "SENSITIVE", f"name matches {pat}", False)

        for pat in _INTERNAL_NAME_PATTERNS:
            if re.search(pat, name_lower):
                return ColumnClassification(name, "INTERNAL", f"identifier-like name matches {pat}", False)

        if self._values_contain_pii(series):
            return ColumnClassification(name, "PII", "sample values match PII patterns", False)

        for pat in _INTERNAL_BUSINESS_PATTERNS:
            if re.search(pat, name_lower):
                return ColumnClassification(name, "INTERNAL", f"business metric name matches {pat}", False)

        if series.dtype == object and series.nunique(dropna=True) > 100:
            return ColumnClassification(name, "INTERNAL", "high-cardinality string", False)

        return ColumnClassification(name, "PUBLIC", "no sensitive signal detected", True)

    def _values_contain_pii(self, series: pd.Series) -> bool:
        if not (series.dtype == object or pd.api.types.is_string_dtype(series)):
            return False
        sample = series.dropna().astype(str).str.strip().head(50)
        pii_hits = 0
        checked = 0
        for val in sample:
            if not val:
                continue
            checked += 1
            # Never classify obvious dates as phone/ID PII.
            if _DATE_LIKE_RE.match(val):
                continue
            compact = re.sub(r"[\s\-()]", "", val)
            if _EMAIL_RE.match(val) or _PAN_RE.match(val) or _AADHAAR_RE.match(compact) or _SSN_RE.match(val):
                pii_hits += 1
            elif _PHONE_RE.match(compact) or _PHONE_SEPARATED_RE.match(val):
                # Indian/mobile-like phone only; avoids short date-like numerics.
                pii_hits += 1
        return checked > 0 and pii_hits / checked >= 0.2

    def safe_columns(self, classifications: dict[str, ColumnClassification]) -> list[str]:
        return [col for col, cls in classifications.items() if cls.safe_for_external]

    def pii_columns(self, classifications: dict[str, ColumnClassification]) -> list[str]:
        return [col for col, cls in classifications.items() if getattr(cls, "level", cls) == "PII"]

    def summary(self, classifications: dict[str, ColumnClassification]) -> dict[str, int]:
        counts = Counter(getattr(cls, "level", str(cls)) for cls in classifications.values())
        for level in ["PUBLIC", "INTERNAL", "CONFIDENTIAL", "PII", "SENSITIVE"]:
            counts.setdefault(level, 0)
        return dict(counts)
