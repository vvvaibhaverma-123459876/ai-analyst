from __future__ import annotations

"""Governed semantic metric registry.

This layer strengthens the earlier metadata.metric_store by normalising metric
truth into a richer object that can be validated, audited, and reused across
planning, execution, and insights.
"""

from dataclasses import dataclass, field
from typing import Any

from core.config import config
from core.exceptions import MetadataError
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class MetricDefinition:
    key: str
    description: str
    aggregation: str = "sum"
    column: str | None = None
    numerator: str | None = None
    denominator: str | None = None
    aliases: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    allowed_grains: list[str] = field(default_factory=list)
    owner: str = "unassigned"
    caveats: list[str] = field(default_factory=list)
    eligible_population: str | None = None
    exclusions: list[str] = field(default_factory=list)
    maturity: str = "draft"
    source_tables: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def formula(self) -> str:
        if self.aggregation == "ratio":
            return f"{self.numerator} / {self.denominator}"
        target = self.column or self.key
        return f"{self.aggregation}({target})"

    def audit_summary(self) -> dict[str, Any]:
        return {
            "metric": self.key,
            "formula": self.formula,
            "owner": self.owner,
            "maturity": self.maturity,
            "grains": self.allowed_grains,
            "dimensions": self.dimensions,
            "eligible_population": self.eligible_population,
            "exclusions": self.exclusions,
            "source_tables": self.source_tables,
            "caveats": self.caveats,
        }


class MetricRegistry:
    def __init__(self, raw_metrics: dict[str, Any] | None = None):
        raw = raw_metrics if raw_metrics is not None else config.METRICS.get("metrics", {})
        self._metrics: dict[str, MetricDefinition] = {}
        self._alias_index: dict[str, str] = {}

        for key, meta in raw.items():
            definition = self._build_definition(key, meta or {})
            self._metrics[key] = definition
            for alias in {key, *(definition.aliases or [])}:
                self._alias_index[alias.lower()] = key

        logger.info("MetricRegistry loaded %d governed metrics.", len(self._metrics))

    def _build_definition(self, key: str, meta: dict[str, Any]) -> MetricDefinition:
        return MetricDefinition(
            key=key,
            description=meta.get("description", key),
            aggregation=meta.get("aggregation", "sum"),
            column=meta.get("column"),
            numerator=meta.get("numerator"),
            denominator=meta.get("denominator"),
            aliases=list(meta.get("aliases", [])),
            dimensions=list(meta.get("dimensions", [])),
            allowed_grains=list(meta.get("allowed_grains", meta.get("grains", ["daily"]))),
            owner=meta.get("owner", "unassigned"),
            caveats=list(meta.get("caveats", [])),
            eligible_population=meta.get("eligible_population"),
            exclusions=list(meta.get("exclusions", [])),
            maturity=meta.get("maturity", "draft"),
            source_tables=list(meta.get("source_tables", [])),
            raw=meta,
        )

    def get(self, name: str) -> MetricDefinition:
        key = self._alias_index.get(name.lower())
        if key is None:
            raise MetadataError(f"Metric not found: '{name}'. Available: {list(self._metrics.keys())}")
        return self._metrics[key]

    def resolve(self, user_text: str) -> str | None:
        lower = user_text.lower()
        matches: list[tuple[int, str]] = []
        for alias, key in self._alias_index.items():
            if alias in lower:
                matches.append((len(alias), key))
        if not matches:
            return None
        matches.sort(reverse=True)
        resolved = matches[0][1]
        logger.info("MetricRegistry resolved '%s' -> '%s'", user_text, resolved)
        return resolved

    def validate_dimension(self, metric_name: str, dimension: str) -> bool:
        metric = self.get(metric_name)
        if not metric.dimensions:
            return True
        return dimension in metric.dimensions

    def validate_grain(self, metric_name: str, grain: str) -> bool:
        metric = self.get(metric_name)
        if not metric.allowed_grains:
            return True
        return grain.lower() in {g.lower() for g in metric.allowed_grains}

    def explain(self, metric_name: str) -> dict[str, Any]:
        return self.get(metric_name).audit_summary()

    def to_prompt_context(self) -> str:
        lines = ["Governed metrics:"]
        for key in self.list_all():
            m = self._metrics[key]
            dims = ", ".join(m.dimensions) if m.dimensions else "any"
            grains = ", ".join(m.allowed_grains) if m.allowed_grains else "any"
            owner = m.owner or "unassigned"
            caveat = f" caveats={'; '.join(m.caveats)}" if m.caveats else ""
            lines.append(
                f"- {m.key}: {m.description} | formula={m.formula} | dims={dims} | grains={grains} | owner={owner} | maturity={m.maturity}.{caveat}"
            )
        return "\n".join(lines)

    def list_all(self) -> list[str]:
        return list(self._metrics.keys())
