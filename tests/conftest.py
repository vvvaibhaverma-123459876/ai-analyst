"""
tests/conftest.py
Shared fixtures and helpers for the entire benchmark suite.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── DataFrame factories ──────────────────────────────────────────────

def make_ts(n=60, kpi="revenue", start="2025-01-01", freq="D",
            trend=1.0, noise=0.05, spike_idx=None):
    """Clean time-series DataFrame with optional spike."""
    dates = pd.date_range(start, periods=n, freq=freq)
    base = np.linspace(100, 100 + trend * n, n)
    vals = base + np.random.default_rng(42).normal(0, noise * base.mean(), n)
    if spike_idx is not None:
        vals[spike_idx] = vals.mean() + 6 * vals.std()
    return pd.DataFrame({"date": dates, kpi: vals})


def make_funnel(n_users=500):
    """Event-level funnel DataFrame: 4 stages."""
    rng = np.random.default_rng(7)
    stages = ["visit", "signup", "onboard", "convert"]
    rows = []
    for uid in range(n_users):
        for i, stage in enumerate(stages):
            if rng.random() < (0.9 ** i):
                rows.append({"user_id": uid, "stage": stage, "date": "2025-06-01"})
    return pd.DataFrame(rows)


def make_cohort(n_users=200):
    """User × activity DataFrame for cohort retention."""
    rng = np.random.default_rng(13)
    rows = []
    for uid in range(n_users):
        signup = pd.Timestamp("2025-01-01") + pd.Timedelta(days=int(rng.integers(0, 30)))
        for d in range(30):
            if rng.random() < max(0.1, 0.9 - d * 0.025):
                rows.append({"user_id": uid,
                             "activity_date": signup + pd.Timedelta(days=d)})
    return pd.DataFrame(rows)


def make_segment(n=300):
    """DataFrame with categorical dimensions for root-cause testing."""
    rng = np.random.default_rng(99)
    channels = rng.choice(["organic", "paid", "referral"], n)
    platforms = rng.choice(["ios", "android", "web"], n)
    revenue = np.where(platforms == "android",
                       rng.normal(40, 8, n),
                       rng.normal(80, 10, n))
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "channel": channels,
        "platform": platforms,
        "revenue": revenue,
    })


def make_ab(n_per_variant=400):
    """A/B experiment DataFrame."""
    rng = np.random.default_rng(55)
    control = pd.DataFrame({
        "variant": "control",
        "converted": rng.binomial(1, 0.10, n_per_variant),
        "revenue": rng.normal(50, 15, n_per_variant),
    })
    treatment = pd.DataFrame({
        "variant": "treatment",
        "converted": rng.binomial(1, 0.13, n_per_variant),
        "revenue": rng.normal(55, 15, n_per_variant),
    })
    return pd.concat([control, treatment], ignore_index=True)


def make_bad_df():
    """DataFrame that should fail the DataQualityGate."""
    return pd.DataFrame({
        "x": [None] * 5 + [1, 2],
        "y": [None] * 7,
        "d": ["bad"] * 7,
    })


def make_empty_df():
    return pd.DataFrame()


# ── Pytest fixtures ──────────────────────────────────────────────────

@pytest.fixture
def ts_df():
    return make_ts()

@pytest.fixture
def ts_df_with_spike():
    return make_ts(n=60, spike_idx=45)

@pytest.fixture
def funnel_df():
    return make_funnel()

@pytest.fixture
def cohort_df():
    return make_cohort()

@pytest.fixture
def segment_df():
    return make_segment()

@pytest.fixture
def ab_df():
    return make_ab()

@pytest.fixture
def bad_df():
    return make_bad_df()
