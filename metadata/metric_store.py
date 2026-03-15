"""
metadata/metric_store.py
Backwards-compatible metric store built on top of the richer semantic registry.
"""

from semantic.metric_registry import MetricRegistry


class MetricStore(MetricRegistry):
    """Compatibility wrapper retained for existing imports."""

    def __init__(self):
        super().__init__()
