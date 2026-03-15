"""
metadata/metric_store.py
Central store for metric definitions loaded from configs/metrics.yaml.
Used by semantic mapper and SQL generator to resolve metric intent.
"""

from core.config import config
from core.exceptions import MetadataError
from core.logger import get_logger

logger = get_logger(__name__)


class MetricStore:

    def __init__(self):
        raw = config.METRICS.get("metrics", {})
        self._metrics: dict = raw
        self._alias_index: dict = self._build_alias_index()
        logger.info(f"MetricStore loaded {len(self._metrics)} metrics.")

    def _build_alias_index(self) -> dict:
        index = {}
        for key, meta in self._metrics.items():
            index[key.lower()] = key
            for alias in meta.get("aliases", []):
                index[alias.lower()] = key
        return index

    def get(self, name: str) -> dict:
        """Retrieve metric definition by exact name or alias."""
        key = self._alias_index.get(name.lower())
        if key is None:
            raise MetadataError(f"Metric not found: '{name}'. Available: {list(self._metrics.keys())}")
        return self._metrics[key]

    def resolve(self, user_text: str) -> str | None:
        """
        Try to find a metric name mentioned in free-form user text.
        Returns metric key or None.
        """
        lower = user_text.lower()
        for alias, key in self._alias_index.items():
            if alias in lower:
                logger.info(f"Metric resolved: '{alias}' → '{key}'")
                return key
        return None

    def to_prompt_context(self) -> str:
        """Returns metric definitions as plain text for LLM prompts."""
        lines = ["Available metrics:"]
        for key, meta in self._metrics.items():
            agg = meta.get("aggregation", "sum")
            if agg == "ratio":
                formula = f"{meta.get('numerator')} / {meta.get('denominator')}"
            else:
                formula = f"{agg}({meta.get('column', key)})"
            lines.append(f"  {key}: {meta['description']} | formula: {formula}")
        return "\n".join(lines)

    def list_all(self) -> list[str]:
        return list(self._metrics.keys())
