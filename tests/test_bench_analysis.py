"""
tests/test_bench_analysis.py  — v9
Benchmark coverage for the analysis/ layer.

v9 update: AnomalyDetector.detect() now returns AnalysisResult (contract-aligned).
Tests use both the new contract interface and the legacy DataFrame interface.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.conftest import make_ts, make_funnel, make_cohort, make_segment
from analysis.anomaly_detector import AnomalyDetector
from analysis.funnel_analyzer import FunnelAnalyzer
from analysis.cohort_analyzer import CohortAnalyzer
from analysis.root_cause import RootCauseAnalyzer
from analysis.statistics import resample_timeseries, period_comparison as compute_period_comparison


# ══════════════════════════════════════════════════════════════════════
# AnomalyDetector — v9 contract interface
# ══════════════════════════════════════════════════════════════════════

class TestAnomalyDetectorContract:

    def test_detect_returns_analysis_result(self):
        from analysis.contract import AnalysisResult
        df = make_ts(n=40)
        result = AnomalyDetector().detect(df, kpi_col="revenue", date_col="date")
        assert isinstance(result, AnalysisResult)
        assert result.module == "anomaly"

    def test_detect_spike_via_contract(self):
        df = make_ts(n=60, spike_idx=45)
        result = AnomalyDetector(z_threshold=2.5).detect(df, kpi_col="revenue", date_col="date")
        assert result.ok
        assert result.anomaly_count >= 1

    def test_kpi_col_alias_value_col(self):
        """value_col must be accepted as alias for kpi_col."""
        df = make_ts(n=40)
        result = AnomalyDetector().detect(df, value_col="revenue", date_col="date")
        assert result.ok

    def test_analyze_contract_method(self):
        """analyze() is the AnalysisContract entry point."""
        df = make_ts(n=40)
        result = AnomalyDetector().analyze(df, kpi_col="revenue", date_col="date")
        assert result.ok

    def test_to_benchmark_output_stable_shape(self):
        df = make_ts(n=40)
        result = AnomalyDetector().detect(df, kpi_col="revenue", date_col="date")
        bm = AnomalyDetector().to_benchmark_output(result)
        for key in ("ok", "module", "anomaly_count", "record_count", "summary",
                    "confidence", "method", "warnings", "errors"):
            assert key in bm

    def test_to_agent_data_backward_compat(self):
        """to_agent_data() must expose keys agents historically expected."""
        df = make_ts(n=40, spike_idx=30)
        result = AnomalyDetector().detect(df, kpi_col="revenue", date_col="date")
        d = result.to_agent_data()
        assert "anomaly_count" in d
        assert "anomaly_records" in d

    def test_missing_col_returns_error_result(self):
        df = make_ts(n=20)
        result = AnomalyDetector().detect(df, kpi_col="nonexistent", date_col="date")
        assert result.ok is False
        assert len(result.errors) >= 1

    def test_empty_df_returns_error_result(self):
        df = pd.DataFrame()
        result = AnomalyDetector().detect(df, kpi_col="revenue", date_col="date")
        assert result.ok is False

    def test_zscore_detects_spike(self):
        df = make_ts(n=60, spike_idx=45)
        result = AnomalyDetector(z_threshold=2.5).detect(df, kpi_col="revenue",
                                                          date_col="date", method="zscore")
        assert result.anomaly_count >= 1

    def test_iqr_method(self):
        df = make_ts(n=60, spike_idx=30)
        result = AnomalyDetector().detect(df, kpi_col="revenue", date_col="date", method="iqr")
        assert result.ok
        assert isinstance(result.anomaly_count, int)

    def test_clean_series_low_false_positives(self):
        df = make_ts(n=60, noise=0.01)
        result = AnomalyDetector(z_threshold=3.0).detect(df, kpi_col="revenue", date_col="date")
        assert result.anomaly_count <= 2

    def test_all_identical_values_zero_anomalies(self):
        df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10), "v": [5.0]*10})
        result = AnomalyDetector().detect(df, kpi_col="v", date_col="date")
        assert result.anomaly_count == 0

    def test_z_threshold_higher_means_fewer_anomalies(self):
        df = make_ts(n=60, noise=0.1, spike_idx=30)
        loose  = AnomalyDetector(z_threshold=1.0).detect(df, "revenue", "date")
        strict = AnomalyDetector(z_threshold=4.0).detect(df, "revenue", "date")
        assert loose.anomaly_count >= strict.anomaly_count

    def test_confidence_between_0_and_1(self):
        df = make_ts(n=40, spike_idx=20)
        result = AnomalyDetector().detect(df, kpi_col="revenue", date_col="date")
        assert 0.0 <= result.confidence <= 1.0


# ══════════════════════════════════════════════════════════════════════
# AnomalyDetector — legacy DataFrame interface (backward compat)
# ══════════════════════════════════════════════════════════════════════

class TestAnomalyDetectorLegacy:

    def test_detect_zscore_returns_dataframe(self):
        df = make_ts(n=40)
        result = AnomalyDetector().detect_zscore(df, date_col="date", value_col="revenue")
        assert isinstance(result, pd.DataFrame)
        assert "anomaly" in result.columns

    def test_detect_zscore_kpi_col_alias(self):
        df = make_ts(n=40)
        result = AnomalyDetector().detect_zscore(df, date_col="date", kpi_col="revenue")
        assert "anomaly" in result.columns

    def test_detect_iqr_returns_dataframe(self):
        df = make_ts(n=40)
        result = AnomalyDetector().detect_iqr(df, value_col="revenue")
        assert isinstance(result, pd.DataFrame)
        assert "anomaly" in result.columns

    def test_detect_iqr_kpi_col_alias(self):
        df = make_ts(n=40)
        result = AnomalyDetector().detect_iqr(df, kpi_col="revenue")
        assert "anomaly" in result.columns


# ══════════════════════════════════════════════════════════════════════
# FunnelAnalyzer
# ══════════════════════════════════════════════════════════════════════

class TestFunnelAnalyzer:

    def test_funnel_conversion_decreasing(self):
        df = make_funnel(n_users=1000)
        result = FunnelAnalyzer().compute_funnel(
            df, stage_col="stage", user_col="user_id",
            stages=["visit","signup","onboard","convert"])
        counts = result["users"].tolist()
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i+1]

    def test_conversion_rates_0_to_100(self):
        df = make_funnel()
        result = FunnelAnalyzer().compute_funnel(df, stage_col="stage", user_col="user_id")
        rates = result["conversion_from_top_pct"].dropna()
        assert (rates >= 0).all() and (rates <= 100).all()

    def test_funnel_returns_all_stages(self):
        df = make_funnel()
        stages = ["visit","signup","onboard","convert"]
        result = FunnelAnalyzer().compute_funnel(
            df, stage_col="stage", user_col="user_id", stages=stages)
        assert set(result["stage"].tolist()) == set(stages)

    def test_empty_df_no_crash(self):
        df = pd.DataFrame({"user_id":[], "stage":[], "date":[]})
        result = FunnelAnalyzer().compute_funnel(df, stage_col="stage", user_col="user_id")
        assert isinstance(result, pd.DataFrame)

    def test_single_stage_no_crash(self):
        df = pd.DataFrame({"user_id":[1,2,3], "stage":["visit","visit","visit"]})
        result = FunnelAnalyzer().compute_funnel(
            df, stage_col="stage", user_col="user_id", stages=["visit"])
        assert len(result) == 1

    def test_dropoff_column_present(self):
        df = make_funnel()
        result = FunnelAnalyzer().compute_funnel(df, stage_col="stage", user_col="user_id")
        assert "drop_off_pct" in result.columns


# ══════════════════════════════════════════════════════════════════════
# CohortAnalyzer
# ══════════════════════════════════════════════════════════════════════

class TestCohortAnalyzer:

    def test_retention_matrix_shape(self):
        df = make_cohort()
        matrix = CohortAnalyzer().build_retention_matrix(
            df, user_col="user_id", date_col="activity_date")
        assert matrix.ndim == 2
        assert matrix.shape[0] >= 1

    def test_retention_values_0_to_100(self):
        df = make_cohort()
        matrix = CohortAnalyzer().build_retention_matrix(
            df, user_col="user_id", date_col="activity_date")
        vals = matrix.values.flatten()
        valid = vals[~np.isnan(vals)]
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_diagonal_100(self):
        df = make_cohort()
        matrix = CohortAnalyzer().build_retention_matrix(
            df, user_col="user_id", date_col="activity_date")
        assert matrix.iloc[0, 0] == pytest.approx(100.0, abs=5)

    def test_retention_decreases(self):
        df = make_cohort()
        matrix = CohortAnalyzer().build_retention_matrix(
            df, user_col="user_id", date_col="activity_date")
        first_row = matrix.iloc[0].dropna().values
        if len(first_row) >= 3:
            assert first_row[0] >= first_row[-1]

    def test_empty_df_no_crash(self):
        df = pd.DataFrame({"user_id":[], "activity_date":[]})
        result = CohortAnalyzer().build_retention_matrix(
            df, user_col="user_id", date_col="activity_date")
        assert isinstance(result, pd.DataFrame)


# ══════════════════════════════════════════════════════════════════════
# RootCauseAnalyzer
# ══════════════════════════════════════════════════════════════════════

class TestRootCauseAnalyzer:

    def test_driver_attribution_returns_movers(self):
        df = make_segment()
        result = RootCauseAnalyzer().driver_attribution(
            df, date_col="date", kpi_col="revenue", days=15)
        movers = result.get("movers", {})
        assert isinstance(movers.get("negative", []), list)
        assert isinstance(movers.get("positive", []), list)

    def test_driver_attribution_required_keys(self):
        df = make_segment()
        result = RootCauseAnalyzer().driver_attribution(
            df, date_col="date", kpi_col="revenue", days=10)
        assert "movers" in result

    def test_empty_df_no_crash(self):
        df = pd.DataFrame({"date":[], "channel":[], "revenue":[]})
        result = RootCauseAnalyzer().driver_attribution(
            df, date_col="date", kpi_col="revenue", days=7)
        assert isinstance(result, dict)

    def test_single_dimension_no_crash(self):
        df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=20),
            "channel": ["organic"] * 20,
            "revenue": np.random.default_rng(1).normal(100, 5, 20),
        })
        result = RootCauseAnalyzer().driver_attribution(
            df, date_col="date", kpi_col="revenue", days=5)
        assert "movers" in result


# ══════════════════════════════════════════════════════════════════════
# Statistics helpers
# ══════════════════════════════════════════════════════════════════════

class TestStatisticsHelpers:

    def test_resample_daily_unchanged(self):
        df = make_ts(n=30)
        result = resample_timeseries(df, "date", "revenue", grain="Daily")
        assert len(result) == 30

    def test_resample_weekly_reduces_rows(self):
        df = make_ts(n=56)
        result = resample_timeseries(df, "date", "revenue", grain="Weekly")
        assert len(result) < 56
        assert len(result) >= 7

    def test_resample_monthly_reduces_rows(self):
        df = make_ts(n=90)
        result = resample_timeseries(df, "date", "revenue", grain="Monthly")
        assert len(result) <= 4

    def test_resample_preserves_sum(self):
        df = make_ts(n=28)
        daily_sum = df["revenue"].sum()
        weekly = resample_timeseries(df, "date", "revenue", grain="Weekly", agg="sum")
        assert abs(weekly["revenue"].sum() - daily_sum) < 0.01

    def test_period_comparison_returns_pct_change(self):
        df = make_ts(n=60)
        result = compute_period_comparison(df, "date", "revenue", comparison="WoW")
        assert isinstance(result, dict)
        assert "pct_change" in result or "change" in result
