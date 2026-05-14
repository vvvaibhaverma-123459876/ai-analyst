"""Local compatibility helpers for the AI Analyst test/runtime environment.

Pandas 2.x interprets `pd.date_range("2025-01-01", 10)` as an `end`
argument. Several legacy tests/notebooks used the older shorthand intending
`periods=10`. This shim keeps that shorthand working inside the project
without changing normal explicit usage.
"""

from __future__ import annotations

try:
    import pandas as _pd

    if not getattr(_pd.date_range, "_ai_analyst_compat", False):
        _orig_date_range = _pd.date_range

        def _compat_date_range(start=None, end=None, periods=None, *args, **kwargs):
            if isinstance(end, int) and periods is None:
                periods = end
                end = None
            return _orig_date_range(start=start, end=end, periods=periods, *args, **kwargs)

        _compat_date_range._ai_analyst_compat = True
        _pd.date_range = _compat_date_range
except Exception:
    pass
