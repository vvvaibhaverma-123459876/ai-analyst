"""
sql/retry_engine.py
Retries SQL generation+validation up to N times, feeding errors back to LLM.
"""

from llm.sql_generator import SQLGenerator
from sql.validator import SQLValidator
from core.exceptions import SQLGenerationError, SQLValidationError
from core.logger import get_logger

logger = get_logger(__name__)

_RETRY_SYSTEM = """You are an expert SQL analyst. 
The previous SQL query had an error. Fix ONLY the error described.
Return ONLY corrected SQL. No explanation. No markdown fences."""


class SQLRetryEngine:

    def __init__(
        self,
        generator: SQLGenerator = None,
        validator: SQLValidator = None,
        max_retries: int = 3,
    ):
        self._generator = generator or SQLGenerator()
        self._validator = validator or SQLValidator()
        self._max_retries = max_retries

    def generate_and_validate(
        self,
        intent: dict,
        schema_context: str,
        metric_context: str,
    ) -> str:
        """
        Attempts SQL generation + validation with automatic retry on failure.
        Returns validated SQL string.
        """
        last_error = None
        sql = None

        for attempt in range(1, self._max_retries + 1):
            logger.info(f"SQL generation attempt {attempt}/{self._max_retries}")
            try:
                if attempt == 1:
                    sql = self._generator.generate(intent, schema_context, metric_context)
                else:
                    # Feed error back to LLM for correction
                    sql = self._retry_with_error(sql, str(last_error))

                validated_sql = self._validator.validate(sql)
                logger.info(f"SQL validated on attempt {attempt}.")
                return validated_sql

            except (SQLGenerationError, SQLValidationError) as e:
                last_error = e
                logger.warning(f"Attempt {attempt} failed: {e}")

        raise SQLGenerationError(
            f"SQL generation failed after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _retry_with_error(self, bad_sql: str, error_msg: str) -> str:
        from llm.client import LLMClient
        llm = LLMClient()
        user = f"Previous SQL:\n{bad_sql}\n\nError:\n{error_msg}\n\nFix the SQL."
        return llm.complete(system=_RETRY_SYSTEM, user=user)
