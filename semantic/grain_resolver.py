from __future__ import annotations

"""Resolve and validate analysis grain against governed metrics."""

from semantic.metric_registry import MetricRegistry


class GrainResolver:
    NORMALISATIONS = {
        "hour": "hourly",
        "hourly": "hourly",
        "day": "daily",
        "daily": "daily",
        "week": "weekly",
        "weekly": "weekly",
        "month": "monthly",
        "monthly": "monthly",
    }

    def __init__(self, registry: MetricRegistry | None = None):
        self.registry = registry or MetricRegistry()

    def normalise(self, grain: str | None) -> str:
        if not grain:
            return "daily"
        return self.NORMALISATIONS.get(grain.strip().lower(), grain.strip().lower())

    def resolve(self, metric_name: str, requested_grain: str | None) -> str:
        grain = self.normalise(requested_grain)
        if self.registry.validate_grain(metric_name, grain):
            return grain
        metric = self.registry.get(metric_name)
        return metric.allowed_grains[0] if metric.allowed_grains else grain
