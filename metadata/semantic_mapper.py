"""
metadata/semantic_mapper.py
Maps user natural language to structured intent:
  metric, dimensions, filters, time_range, analysis_type.
Uses glossary + metric store first; LLM as fallback.
"""

import re
from core.config import config
from core.logger import get_logger
from metadata.metric_store import MetricStore

logger = get_logger(__name__)

GLOSSARY = config.GLOSSARY


class SemanticMapper:

    def __init__(self, metric_store: MetricStore = None):
        self._metric_store = metric_store or MetricStore()
        self._time_expressions = GLOSSARY.get("time_expressions", {})
        self._segments = GLOSSARY.get("segments", {})
        self._funnel_stages = GLOSSARY.get("funnel_stages", [])

    def interpret(self, user_question: str, schema_context: dict = None) -> dict:
        """
        Returns structured intent dict:
        {
            metric: str | None,
            dimensions: list[str],
            filters: dict,
            time_range: str | None,
            analysis_type: str,
            raw_question: str,
        }
        """
        q = user_question.lower()

        metric = self._metric_store.resolve(q)
        time_range = self._resolve_time(q)
        filters = self._resolve_filters(q)
        analysis_type = self._resolve_analysis_type(q)
        dimensions = self._resolve_dimensions(q, schema_context)

        intent = {
            "metric": metric,
            "dimensions": dimensions,
            "filters": filters,
            "time_range": time_range,
            "analysis_type": analysis_type,
            "raw_question": user_question,
        }

        logger.info(f"Intent resolved: {intent}")
        return intent

    def _resolve_time(self, q: str) -> str | None:
        for phrase, expr in self._time_expressions.items():
            if phrase.replace("_", " ") in q or phrase in q:
                return expr
        if re.search(r"\d{4}-\d{2}-\d{2}", q):
            return re.search(r"\d{4}-\d{2}-\d{2}", q).group()
        return None

    def _resolve_filters(self, q: str) -> dict:
        filters = {}
        for seg_name, seg_def in self._segments.items():
            if seg_name in q:
                col = seg_def["column"]
                val = seg_def["value"]
                filters[col] = val
        return filters

    def _resolve_analysis_type(self, q: str) -> str:
        if any(w in q for w in ["drop", "fall", "decline", "down", "decrease"]):
            return "root_cause"
        if any(w in q for w in ["spike", "jump", "surge", "up", "increase"]):
            return "root_cause"
        if any(w in q for w in ["funnel", "conversion", "drop-off", "dropoff"]):
            return "funnel"
        if any(w in q for w in ["cohort", "retention", "d1", "d7", "d30"]):
            return "cohort"
        if any(w in q for w in ["anomal", "unusual", "weird", "unexpected"]):
            return "anomaly"
        if any(w in q for w in ["trend", "over time", "by day", "by week", "by month"]):
            return "trend"
        if any(w in q for w in ["why", "cause", "driver", "reason"]):
            return "driver_attribution"
        return "trend"

    def _resolve_dimensions(self, q: str, schema_context: dict = None) -> list[str]:
        dims = []
        if schema_context:
            for table_meta in schema_context.values():
                for col in table_meta.get("columns", []):
                    col_name = col["name"] if isinstance(col, dict) else col
                    if col_name.lower() in q:
                        dims.append(col_name)
        return dims
