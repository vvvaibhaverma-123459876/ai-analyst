from __future__ import annotations

from security.pii_masker import PIIMasker


class RedactionEngine:
    """Prompt/output redaction wrapper.

    Delegates pattern handling to PIIMasker so the system has a single PII
    redaction implementation instead of parallel regex stacks.
    """

    def __init__(self, masker: PIIMasker | None = None):
        self._masker = masker or PIIMasker()

    def redact(self, text: str) -> str:
        return self._masker.mask_prompt(text or '')
