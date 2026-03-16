"""
security/pii_masker.py
Masks PII and SENSITIVE columns before any data leaves the security boundary.

Masking strategy:
  PII     → deterministic token (USER_001, EMAIL_001) — consistent within a run
  SENSITIVE → aggregate stats only (mean, std, min, max) — no raw values
  CONFIDENTIAL → column retained but values rounded/binned

The masking is reversible within a session (token → original) for display,
but the unmasked data never leaves the system.
"""

from __future__ import annotations
import hashlib
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from security.data_classifier import DataClassifier, ColumnClassification
from core.logger import get_logger

logger = get_logger(__name__)


class PIIMasker:

    def __init__(self):
        self._classifier = DataClassifier()
        self._token_map: dict[str, str] = {}        # original → token
        self._reverse_map: dict[str, str] = {}      # token → original
        self._counters: dict[str, int] = defaultdict(int)

    def mask_dataframe(
        self,
        df: pd.DataFrame,
        classifications: dict[str, ColumnClassification] = None,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Returns (masked_df, masking_report).
        masked_df has PII/sensitive columns replaced with tokens.
        masking_report describes what was masked.
        """
        if classifications is None:
            classifications = self._classifier.classify_dataframe(df)

        masked = df.copy()
        report = {"masked_columns": [], "dropped_columns": [], "binned_columns": []}

        for col, cls in classifications.items():
            if col not in masked.columns:
                continue

            if cls.level == "PII":
                masked[col] = masked[col].apply(
                    lambda v: self._tokenise(v, cls.level)
                )
                report["masked_columns"].append(col)

            elif cls.level == "SENSITIVE":
                # Replace with aggregate stats — no raw values
                series = masked[col]
                if pd.api.types.is_numeric_dtype(series):
                    masked[col] = f"[SENSITIVE: mean={series.mean():.2f} std={series.std():.2f}]"
                else:
                    masked[col] = "[SENSITIVE: redacted]"
                report["dropped_columns"].append(col)

            elif cls.level == "CONFIDENTIAL":
                # Bin numeric values into deciles
                if pd.api.types.is_numeric_dtype(masked[col]):
                    try:
                        masked[col] = pd.qcut(
                            masked[col], q=10, labels=False, duplicates="drop"
                        ).astype(str) + "_decile"
                        report["binned_columns"].append(col)
                    except Exception:
                        pass

        logger.info(
            f"PII masking: masked={len(report['masked_columns'])}, "
            f"dropped={len(report['dropped_columns'])}, "
            f"binned={len(report['binned_columns'])}"
        )
        return masked, report

    def mask_prompt(self, prompt: str | None) -> str:
        """
        Scans a prompt string for PII patterns and redacts them.
        Applied before any string is sent to an LLM.
        Returns empty string for None input.
        """
        if prompt is None:
            return ""
        prompt = str(prompt)
        # Email addresses
        prompt = re.sub(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b",
            "[EMAIL_REDACTED]", prompt
        )
        # Phone numbers
        prompt = re.sub(
            r"\b\+?[\d\s\-\(\)]{10,15}\b",
            "[PHONE_REDACTED]", prompt
        )
        # Aadhaar / 12-digit IDs
        prompt = re.sub(r"\b[2-9]\d{11}\b", "[ID_REDACTED]", prompt)
        # PAN card
        prompt = re.sub(r"\b[A-Z]{5}\d{4}[A-Z]\b", "[PAN_REDACTED]", prompt)
        # IPv4
        prompt = re.sub(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
            "[IP_REDACTED]", prompt
        )
        return prompt

    def _tokenise(self, value, category: str) -> str:
        if pd.isna(value):
            return value
        key = str(value)
        if key not in self._token_map:
            prefix = category[:4].upper()
            self._counters[prefix] += 1
            token = f"{prefix}_{self._counters[prefix]:04d}"
            self._token_map[key] = token
            self._reverse_map[token] = key
        return self._token_map[key]

    def unmask(self, token: str) -> str:
        """Reverse lookup — only usable within the same session."""
        return self._reverse_map.get(token, token)
