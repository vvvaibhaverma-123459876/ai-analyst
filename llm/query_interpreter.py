"""
llm/query_interpreter.py
Wraps SemanticMapper. Falls back to LLM if rule-based mapping is incomplete.
"""

from llm.client import LLMClient
from metadata.semantic_mapper import SemanticMapper
from core.logger import get_logger
import json

logger = get_logger(__name__)

_INTERPRET_SYSTEM = """You are an analytics query interpreter.
Given a natural language question, extract structured intent as JSON only.
Output format:
{
  "metric": "<metric name or null>",
  "dimensions": ["<col1>", "<col2>"],
  "filters": {"<col>": "<value>"},
  "time_range": "<time expression or null>",
  "analysis_type": "trend|anomaly|driver_attribution|root_cause|funnel|cohort"
}
Return only valid JSON. No explanation."""


class QueryInterpreter:

    def __init__(self, semantic_mapper: SemanticMapper = None, llm_client: LLMClient = None):
        self._mapper = semantic_mapper or SemanticMapper()
        self._llm = llm_client or LLMClient()

    def interpret(self, question: str, schema_context: dict = None) -> dict:
        # Step 1: rule-based fast path
        intent = self._mapper.interpret(question, schema_context)

        # Step 2: if metric still unresolved, try LLM
        if intent["metric"] is None:
            logger.info("Metric not resolved by rules — attempting LLM fallback.")
            intent = self._llm_interpret(question, schema_context, intent)

        return intent

    def _llm_interpret(self, question: str, schema_context: dict, base_intent: dict) -> dict:
        schema_str = str(schema_context) if schema_context else "Not available"
        user_prompt = f"Schema context:\n{schema_str}\n\nUser question: {question}"
        try:
            raw = self._llm.complete(system=_INTERPRET_SYSTEM, user=user_prompt)
            # strip fences
            raw = raw.strip().strip("```json").strip("```").strip()
            parsed = json.loads(raw)
            # merge: LLM result fills gaps in rule-based result
            for k, v in parsed.items():
                if not base_intent.get(k):
                    base_intent[k] = v
            logger.info(f"LLM-enriched intent: {base_intent}")
        except Exception as e:
            logger.warning(f"LLM intent parsing failed, using rule-based result: {e}")
        return base_intent
