"""
sql/validator.py
Safety and format checks for LLM-generated SQL.
Blocks destructive statements. Enforces row limits.
"""

import re
from core.constants import SAFE_SQL_KEYWORDS, UNSAFE_SQL_KEYWORDS
from core.exceptions import SQLValidationError
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)


class SQLValidator:

    def __init__(self, max_rows: int = None):
        self._max_rows = max_rows or config.MAX_QUERY_ROWS

    def validate(self, sql: str) -> str:
        """
        Validates SQL string. Returns cleaned SQL on success.
        Raises SQLValidationError on any violation.
        """
        if not sql or not sql.strip():
            raise SQLValidationError("SQL is empty.")

        cleaned = sql.strip().rstrip(";")
        upper = cleaned.upper()

        # Must start with SELECT or WITH
        first_word = upper.split()[0]
        if first_word not in SAFE_SQL_KEYWORDS:
            raise SQLValidationError(
                f"SQL must start with SELECT or WITH. Got: '{first_word}'"
            )

        # Block destructive keywords
        for kw in UNSAFE_SQL_KEYWORDS:
            pattern = rf"\b{kw}\b"
            if re.search(pattern, upper):
                raise SQLValidationError(
                    f"Unsafe keyword detected: '{kw}'. Only SELECT queries are allowed."
                )

        # Warn if no LIMIT — inject one
        if "LIMIT" not in upper:
            cleaned = f"{cleaned}\nLIMIT {self._max_rows}"
            logger.warning(f"No LIMIT found — injected LIMIT {self._max_rows}")

        # Block multiple statements
        statements = [s.strip() for s in cleaned.split(";") if s.strip()]
        if len(statements) > 1:
            raise SQLValidationError("Multiple SQL statements are not allowed.")

        logger.info("SQL validation passed.")
        return cleaned
