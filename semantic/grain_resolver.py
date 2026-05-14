from __future__ import annotations

"""Resolve and validate analysis grain against governed metrics."""

from semantic.metric_registry import MetricRegistry


class GrainResolver:
    NORMALISATIONS = {
        "hour": "daily",      # default downgrade; hourly often unsupported unless explicitly governed
        "hourly": "hourly",
        "day": "daily",
        "daily": "daily",
        "week": "weekly",
        "weekly": "weekly",
        "month": "monthly",
        "monthly": "monthly",
    }
    DISPLAY = {"hourly": "Hourly", "daily": "Daily", "weekly": "Weekly", "monthly": "Monthly"}

    def __init__(self, registry: MetricRegistry | None = None):
        self.registry = registry or MetricRegistry()

    def normalise(self, grain: str | None) -> str:
        if not grain:
            return "daily"
        return self.NORMALISATIONS.get(str(grain).strip().lower(), str(grain).strip().lower())

    def _display(self, grain: str) -> str:
        return self.DISPLAY.get(str(grain).lower(), str(grain).title())

    def resolve(self, metric_name: str, requested_grain: str | None) -> str:
        grain = self.normalise(requested_grain)
        try:
            if self.registry.validate_grain(metric_name, grain):
                return self._display(grain)
            metric = self.registry.get(metric_name)
            fallback = metric.allowed_grains[0] if metric.allowed_grains else grain
            return self._display(fallback)
        except Exception:
            # Unknown metric: keep a safe, display-normalised default rather than raising.
            return self._display(grain or "daily")
