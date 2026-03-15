"""
llm/sql_generator.py
Generates SQL from structured intent using LLM + schema context.
"""

from llm.client import LLMClient
from llm.prompts import Prompts
from core.exceptions import SQLGenerationError
from core.logger import get_logger

logger = get_logger(__name__)


class SQLGenerator:

    def __init__(self, llm_client: LLMClient = None):
        self._llm = llm_client or LLMClient()

    def generate(
        self,
        intent: dict,
        schema_context: str,
        metric_context: str,
    ) -> str:
        """
        Returns a raw SQL string.
        Raises SQLGenerationError if LLM returns nothing useful.
        """
        user_prompt = Prompts.sql_user(intent, schema_context, metric_context)
        try:
            sql = self._llm.complete(
                system=Prompts.SQL_SYSTEM,
                user=user_prompt,
            )
        except Exception as e:
            raise SQLGenerationError(f"LLM failed to generate SQL: {e}") from e

        sql = sql.strip()
        if not sql:
            raise SQLGenerationError("LLM returned empty SQL.")

        # Strip markdown fences if model ignored instructions
        if sql.startswith("```"):
            lines = sql.splitlines()
            sql = "\n".join(
                line for line in lines
                if not line.strip().startswith("```")
            ).strip()

        logger.info(f"Generated SQL:\n{sql}")
        return sql
