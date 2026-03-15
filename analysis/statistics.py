"""
analysis/statistics.py
Time series helpers: resampling, trend line, MoM/WoW/DoD comparisons.
"""

import pandas as pd
import numpy as np
from core.constants import RESAMPLE_MAP
from core.logger import get_logger

logger = get_logger(__name__)


def resample_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    grain: str = "Daily",
    agg: str = "sum",
) -> pd.DataFrame:
    """Resample a time series to the given grain. Preserved from app.py."""
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.groupby(date_col)[value_col].agg(agg).reset_index()

    freq = RESAMPLE_MAP.get(grain)
    if freq and grain != "Daily":
        ts = ts.set_index(date_col).resample(freq)[value_col].agg(agg).reset_index()

    return ts


def period_comparison(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    comparison: str = "WoW",
) -> dict:
    """
    Compare the most recent period against the equivalent prior period.
    comparison: 'DoD' | 'WoW' | 'MoM'
    """
    period_map = {"DoD": 1, "WoW": 7, "MoM": 30}
    days = period_map.get(comparison, 7)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    max_date = df[date_col].max()
    current_start = max_date - pd.Timedelta(days=days - 1)
    prior_start = current_start - pd.Timedelta(days=days)

    current = df[df[date_col] >= current_start][value_col].sum()
    prior = df[(df[date_col] >= prior_start) & (df[date_col] < current_start)][value_col].sum()

    delta = current - prior
    pct = (delta / prior * 100) if prior != 0 else 0

    return {
        "comparison": comparison,
        "current": round(current, 2),
        "prior": round(prior, 2),
        "delta": round(delta, 2),
        "pct_change": round(pct, 2),
    }


def add_trend_line(ts: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """Adds a linear trend column to a time series DataFrame."""
    result = ts.copy()
    x = np.arange(len(result))
    y = result[value_col].fillna(0).values
    if len(x) >= 2:
        coeffs = np.polyfit(x, y, 1)
        result["trend"] = np.polyval(coeffs, x)
    else:
        result["trend"] = y
    return result


def rolling_stats(
    ts: pd.DataFrame,
    value_col: str,
    window: int = 7,
) -> pd.DataFrame:
    """Adds rolling mean, std, and upper/lower bands."""
    result = ts.copy()
    result["rolling_mean"] = result[value_col].rolling(window).mean()
    result["rolling_std"] = result[value_col].rolling(window).std()
    result["upper_band"] = result["rolling_mean"] + 2 * result["rolling_std"]
    result["lower_band"] = result["rolling_mean"] - 2 * result["rolling_std"]
    return result
