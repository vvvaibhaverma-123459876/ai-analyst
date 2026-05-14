from __future__ import annotations

import re
from typing import Any


class OutputClassifier:
    """Classifies outbound payloads so routers/API responses can redact or audit."""

    _PII_PATTERNS = [
        re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
        re.compile(r'\b(?:\+?91[-\s]?)?[6-9]\d{9}\b'),
        re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    ]
    _REDACTED_PATTERNS = [
        re.compile(r'\[[A-Z_ ]*REDACTED[A-Z_ ]*\]', re.I),
        re.compile(r'\[REDACTED\s+[^\]]+\]', re.I),
        re.compile(r'\[[A-Z_ ]*MASKED[A-Z_ ]*\]', re.I),
    ]

    def classify(self, payload: Any) -> str:
        text_original = self._flatten(payload)
        text = text_original.lower()
        if any(p.search(text_original) for p in self._REDACTED_PATTERNS):
            return 'RESTRICTED'
        if any(word in text for word in ('policy block', 'tenant isolation', 'access denied')):
            return 'RESTRICTED'
        if any(p.search(text_original) for p in self._PII_PATTERNS):
            return 'CONFIDENTIAL'
        return 'INTERNAL'

    def _flatten(self, payload: Any) -> str:
        if payload is None:
            return ''
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return ' '.join(self._flatten(v) for v in payload.values())
        if isinstance(payload, (list, tuple, set)):
            return ' '.join(self._flatten(v) for v in payload)
        return str(payload)
