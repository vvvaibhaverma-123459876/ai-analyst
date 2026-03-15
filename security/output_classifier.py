from __future__ import annotations

import re
from typing import Any


class OutputClassifier:
    """Classifies outbound payloads so routers/API responses can redact or audit."""

    _PII_PATTERNS = [
        re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
        re.compile(r'\b\d{10}\b'),
        re.compile(r'\b[A-Z]{5}\d{4}[A-Z]\b'),
        re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    ]

    def classify(self, payload: Any) -> str:
        text = self._flatten(payload).lower()
        if '[redacted' in text or 'masked' in text:
            return 'RESTRICTED'
        if any(p.search(text) for p in self._PII_PATTERNS):
            return 'CONFIDENTIAL'
        if any(word in text for word in ('policy block', 'tenant isolation', 'access denied')):
            return 'RESTRICTED'
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
