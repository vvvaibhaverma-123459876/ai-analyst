"""
sql/validator.py
Safety and format checks for LLM-generated SQL.

Public contract:
- validate(sql) -> list[str] of validation errors. Empty list means safe.
- sanitize(sql) -> cleaned SELECT/WITH SQL with LIMIT injection or raises.
- validate_or_raise(sql) -> alias for sanitize() for old production call-sites.
"""

from __future__ import annotations

import re
from core.constants import SAFE_SQL_KEYWORDS, UNSAFE_SQL_KEYWORDS
from core.exceptions import SQLValidationError
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)


class SQLValidator:
    def __init__(self, max_rows: int = None):
        self._max_rows = max_rows or config.MAX_QUERY_ROWS

    def validate(self, sql: str) -> list[str]:
        errors: list[str] = []
        if not sql or not str(sql).strip():
            return ["SQL is empty."]

        cleaned = str(sql).strip().rstrip(";")
        upper = cleaned.upper()
        tokens = upper.split()
        if not tokens:
            return ["SQL is empty."]

        first_word = tokens[0]
        if first_word not in SAFE_SQL_KEYWORDS:
            errors.append(f"SQL must start with SELECT or WITH. Got: '{first_word}'.")

        for kw in UNSAFE_SQL_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", upper):
                errors.append(f"Unsafe/dangerous keyword detected: '{kw}'. Only SELECT queries are allowed.")

        statements = [s.strip() for s in cleaned.split(";") if s.strip()]
        if len(statements) > 1:
            errors.append("Multiple SQL statements are not allowed.")
        return errors

    def sanitize(self, sql: str) -> str:
        errors = self.validate(sql)
        if errors:
            raise SQLValidationError("; ".join(errors))
        cleaned = str(sql).strip().rstrip(";")
        if "LIMIT" not in cleaned.upper():
            cleaned = f"{cleaned}\nLIMIT {self._max_rows}"
            logger.warning("No LIMIT found — injected LIMIT %s", self._max_rows)
        logger.info("SQL validation passed.")
        return cleaned

    def validate_or_raise(self, sql: str) -> str:
        return self.sanitize(sql)
