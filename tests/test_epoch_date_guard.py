"""
tests/test_epoch_date_guard.py

Reproduces and locks in the fix for bug 3: the KPI trend x-axis rendered
"Dec 31 1969 23:59:59.999 / Jan 1 1970" — a numeric column (a 0/1 flag) was
coerced with pd.to_datetime on raw ints, which pandas silently reinterprets
as nanoseconds-since-epoch, collapsing 18,965 rows to a single/near-single
time bucket. That fed "forecast · skipped: Only 1 points — need >=10" and,
worse, a fabricated-looking trend chart instead of a clear error.

DataQualityGate.assess() now detects this exact signature (numeric source
column + parsed dates collapsing to a degenerate range) and reports it via a
new `fatal_reasons` field, distinct from the existing `blocking_reasons`
(which lower confidence but let the pipeline continue). GovernedPipeline.run()
raises DataQualityError and halts when fatal_reasons is non-empty — the
pipeline must not proceed with garbage.
"""
import pandas as pd
import pytest

from quality.data_quality_gate import DataQualityGate
from core.exceptions import DataQualityError


def test_numeric_flag_column_as_date_is_fatal():
    """The exact reported scenario: a 0/1 flag column chosen as the date column."""
    df = pd.DataFrame({
        "vkyc_success_event": ([0, 1] * 5000)[:9483] + ([0] * 9482),
        "payment_success_event": [1] * 2530 + [0] * 16435,
    })
    dq = DataQualityGate().assess(df, "vkyc_success_event", "payment_success_event")
    assert dq.ok is False
    assert dq.fatal_reasons, "a numeric flag column parsed as a date must be fatal, not just a soft blocking reason"
    assert "not a date column" in dq.fatal_reasons[0]


def test_constant_numeric_column_as_date_is_fatal():
    """A constant (nunique=1) numeric column across a multi-thousand-row
    frame — e.g. event_count, always 1 — is the other degenerate signature."""
    df = pd.DataFrame({
        "event_count": [1] * 5000,
        "amount": range(5000),
    })
    dq = DataQualityGate().assess(df, "event_count", "amount")
    assert dq.ok is False
    assert dq.fatal_reasons


def test_real_date_column_is_not_flagged_fatal():
    """Sanity check: a genuine date column (even numeric-looking as a
    string, or already datetime64) must never trigger the fatal path."""
    df = pd.DataFrame({
        "event_time": pd.date_range("2026-01-01", periods=200, freq="D").astype(str),
        "amount": range(200),
    })
    dq = DataQualityGate().assess(df, "event_time", "amount")
    assert not dq.fatal_reasons

    df2 = pd.DataFrame({
        "event_time": pd.date_range("2026-01-01", periods=200, freq="D"),
        "amount": range(200),
    })
    dq2 = DataQualityGate().assess(df2, "event_time", "amount")
    assert not dq2.fatal_reasons


def test_numeric_nanosecond_epoch_is_fine_but_second_epoch_is_caught():
    """pd.to_datetime with no explicit unit always treats a bare integer
    Series as nanoseconds since epoch (this codebase never passes unit=),
    so genuine epoch-*nanosecond* values round-trip correctly and must not
    be flagged — while epoch-*seconds* values (a ~10-digit int, the far more
    common real-world representation) get misinterpreted as a handful of
    nanoseconds after 1970-01-01 and are correctly caught as degenerate;
    this is exactly the class of silent misparse the guard exists for, not
    a false positive."""
    df_ns = pd.DataFrame({
        "unix_ts_ns": pd.date_range("2020-01-01", periods=500, freq="h").astype("int64"),
        "amount": range(500),
    })
    dq_ns = DataQualityGate().assess(df_ns, "unix_ts_ns", "amount")
    assert not dq_ns.fatal_reasons

    df_s = pd.DataFrame({
        "unix_ts_s": pd.date_range("2020-01-01", periods=500, freq="h").astype("int64") // 10**9,
        "amount": range(500),
    })
    dq_s = DataQualityGate().assess(df_s, "unix_ts_s", "amount")
    assert dq_s.fatal_reasons


def test_governed_pipeline_raises_dataqualityerror_instead_of_producing_a_chart():
    from agents.context import AnalysisContext
    from agents.pipeline import GovernedPipeline

    df = pd.DataFrame({
        "vkyc_success_event": ([0, 1] * 5000)[:9483] + ([0] * 9482),
        "payment_success_event": [1] * 2530 + [0] * 16435,
    })
    ctx = AnalysisContext(
        df=df, date_col="vkyc_success_event", kpi_col="payment_success_event",
        tenant_id="default", user_id="test",
    )
    with pytest.raises(DataQualityError, match="not a date column"):
        GovernedPipeline().run(ctx)


def test_real_fintech_demo_dataset_reproduces_and_is_caught():
    """The exact real-world case from the observed bug report."""
    from pathlib import Path

    csv_path = Path(__file__).resolve().parent.parent / "sample_fintech_onboarding.csv"
    df = pd.read_csv(csv_path)
    dq = DataQualityGate().assess(df, "vkyc_success_event", "payment_success_event")
    assert dq.fatal_reasons
    assert "vkyc_success_event" in dq.fatal_reasons[0]

    # And the same real column, correctly chosen (event_time — see bug 2's
    # fix), must NOT be flagged.
    dq_correct = DataQualityGate().assess(df, "event_time", "payment_success_event")
    assert not dq_correct.fatal_reasons
