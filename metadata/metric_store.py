"""
metadata/metric_store.py  — v9  DEPRECATED

MetricStore is now a thin forwarding wrapper over semantic.MetricRegistry.
Do not add new logic here.  All callers should import MetricRegistry directly.

This file is retained ONLY to avoid breaking existing imports while the
migration completes.  It will be removed in v10.
"""
from __future__ import annotations
import warnings
from semantic.metric_registry import MetricRegistry   # single source of truth


class MetricStore(MetricRegistry):
    """
    DEPRECATED — use semantic.metric_registry.MetricRegistry directly.
    MetricStore is a zero-logic subclass kept for import compatibility.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "MetricStore is deprecated. Import MetricRegistry from "
            "semantic.metric_registry instead. MetricStore will be removed in v10.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
