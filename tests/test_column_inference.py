"""
tests/test_column_inference.py

Reproduces and locks in the fix for bug 2: on the bundled fintech onboarding
dataset, the setup screen auto-selected Date column = vkyc_success_event (a
numeric 0/1 flag column) and Funnel stage = event_time (a timestamp column).

Root cause: pd.to_datetime silently reinterprets raw integers as
nanoseconds-since-epoch, so the old heuristic's "does it parse as a date"
check spuriously succeeded on numeric flag/count columns, and any column
whose name merely contained "event" (event_time, activation_event,
vkyc_success_event, payment_success_event, event_count) got a date-name-hint
bonus regardless of whether it was actually a date.

Needs streamlit (app/ui/product_app.py imports it at module level) — skipped
under the lean requirements-ci.txt env, covered by the demo-smoke CI job
which installs requirements-demo.txt.
"""
import pandas as pd
import pytest

pytest.importorskip("streamlit")

from app.ui.product_app import ROOT, guess_columns, resolve_columns, load_sample  # noqa: E402


def test_fintech_dataset_column_inference_is_correct():
    df = load_sample("Fintech onboarding demo")
    guess = resolve_columns(df, "sample_fintech_onboarding.csv")

    assert guess.date_col == "event_time"
    assert guess.kpi_col == "payment_success_event"
    assert guess.user_col == "user_id"
    assert guess.stage_col == "stage"
    assert set(guess.dimensions) <= {"channel", "platform", "city", "app_version"}


def test_growth_dataset_column_inference_is_correct():
    df = load_sample("Growth KPI demo")
    guess = resolve_columns(df, "sample_growth_data.csv")

    assert guess.date_col == "date"
    assert guess.kpi_col == "revenue"
    assert guess.stage_col is None, "growth dataset has no funnel concept — must not invent a stage column"
    assert set(guess.dimensions) <= {"channel", "region", "device"}


def test_heuristic_never_picks_a_numeric_column_as_date():
    """Without the per-dataset default (i.e. the pure heuristic fallback
    path), a numeric flag/count column must never win date candidacy just
    because its name contains a date-ish substring like "event"."""
    df = pd.DataFrame({
        "vkyc_success_event": [0, 1, 0, 1, 0] * 20,
        "event_count": [1] * 100,
        "event_time": pd.date_range("2026-01-01", periods=100, freq="D").astype(str),
        "amount": range(100),
    })
    guess = guess_columns(df)
    assert guess.date_col == "event_time"


def test_heuristic_never_picks_a_timestamp_as_funnel_stage():
    df = pd.DataFrame({
        "event_time": pd.date_range("2026-01-01", periods=60, freq="D").astype(str),
        "stage": (["Signup", "Verified", "Activated"] * 20),
        "amount": range(60),
    })
    guess = guess_columns(df)
    assert guess.date_col == "event_time"
    assert guess.stage_col == "stage"


def test_dataset_default_mapping_columns_all_exist_in_bundled_csvs():
    """Guards against DATASET_DEFAULT_COLUMNS drifting out of sync with the
    actual bundled CSVs (e.g. if a column gets renamed later)."""
    from app.ui.product_app import DATASET_DEFAULT_COLUMNS

    for filename, mapping in DATASET_DEFAULT_COLUMNS.items():
        df = pd.read_csv(ROOT / filename, nrows=5)
        for col in (mapping.date_col, mapping.kpi_col, mapping.user_col, mapping.stage_col):
            if col:
                assert col in df.columns, f"{filename}: mapped column {col!r} not found"
        for col in mapping.dimensions:
            assert col in df.columns, f"{filename}: mapped dimension {col!r} not found"
