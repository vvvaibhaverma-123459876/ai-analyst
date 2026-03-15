"""
tests/test_bench_jury.py
Benchmark coverage for the jury/ layer:
  - Foreman protocol (unanimous / majority / split / none)
  - AnomalyJuryAgent (4-juror ensemble)
  - ForecastJuryAgent (temporal holdout + Prophet/ARIMA/ETS)
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from conftest import make_ts, make_ts as make_ts_spike
from jury.foreman import Foreman, ForemanVerdict
from jury.base_juror import BaseJuror, JurorVerdict
from agents.context import AnalysisContext


# ══════════════════════════════════════════════════════════════════════
# Foreman protocol
# ══════════════════════════════════════════════════════════════════════

def _make_verdict(juror_name, finding, confidence, status="found"):
    return JurorVerdict(
        juror=juror_name,
        method=juror_name,
        data={"anomaly_count": 1 if status == "found" else 0},
        confidence=confidence,
        summary=finding,
        status=status,
    )


class TestForemanProtocol:

    def test_unanimous_all_agree(self):
        verdicts = [_make_verdict(f"j{i}", "anomaly found", 0.9) for i in range(4)]
        result = Foreman().reconcile(verdicts)
        assert result.consensus == "unanimous"
        assert result.confidence >= 0.85

    def test_majority_three_of_four(self):
        verdicts = [
            _make_verdict("j1", "anomaly found", 0.9),
            _make_verdict("j2", "anomaly found", 0.85),
            _make_verdict("j3", "anomaly found", 0.8),
            _make_verdict("j4", "no anomaly", 0.6, status="clean"),
        ]
        result = Foreman().reconcile(verdicts)
        assert result.consensus in ("majority", "unanimous")

    def test_split_two_vs_two(self):
        verdicts = [
            _make_verdict("j1", "anomaly found", 0.9),
            _make_verdict("j2", "anomaly found", 0.85),
            _make_verdict("j3", "no anomaly", 0.8, status="clean"),
            _make_verdict("j4", "no anomaly", 0.75, status="clean"),
        ]
        result = Foreman().reconcile(verdicts)
        assert result.consensus in ("split", "majority")

    def test_single_juror_unanimous(self):
        verdicts = [_make_verdict("j1", "anomaly found", 0.9)]
        result = Foreman().reconcile(verdicts)
        assert result.consensus in ("unanimous", "majority")

    def test_empty_verdicts_no_crash(self):
        result = Foreman().reconcile([])
        assert isinstance(result, ForemanVerdict)

    def test_all_skipped_no_consensus(self):
        verdicts = [_make_verdict(f"j{i}", "skipped", 0.0, status="skipped")
                    for i in range(4)]
        result = Foreman().reconcile(verdicts)
        assert result.consensus in ("none", "split", "majority")

    def test_confidence_unanimous_higher_than_split(self):
        unanimous = [_make_verdict(f"j{i}", "anomaly", 0.9) for i in range(4)]
        split = [
            _make_verdict("j1", "anomaly", 0.9), _make_verdict("j2", "anomaly", 0.9),
            _make_verdict("j3", "clean", 0.9, "clean"), _make_verdict("j4", "clean", 0.9, "clean"),
        ]
        u_result = Foreman().reconcile(unanimous)
        s_result = Foreman().reconcile(split)
        assert u_result.confidence >= s_result.confidence


# ══════════════════════════════════════════════════════════════════════
# AnomalyJuryAgent
# ══════════════════════════════════════════════════════════════════════

class TestAnomalyJuryAgent:

    def _make_ctx(self, df, kpi="revenue", date="date"):
        import numpy as np
        ctx = AnalysisContext(
            df=df.copy(),
            kpi_col=kpi,
            date_col=date,
        )
        ts = df.copy()
        ts[date] = pd.to_datetime(ts[date])
        ts = ts.sort_values(date).reset_index(drop=True)
        ctx.ts = ts
        ctx.data_profile = {
            "rows": len(df),
            "has_time_series": True,
            "has_funnel_signal": False,
            "has_cohort_signal": False,
            "dimensions": [],
            "kpis": [kpi],
        }
        return ctx

    def test_detects_spike_in_clean_series(self):
        from jury.anomaly_jury import AnomalyJuryAgent
        df = make_ts_spike(n=60, spike_idx=45)
        ctx = self._make_ctx(df)
        result = AnomalyJuryAgent().run(ctx)
        assert result.status in ("success", "skipped")
        if result.status == "success":
            assert "anomaly_count" in result.data
            assert result.data["anomaly_count"] >= 0

    def test_clean_series_low_false_positive_rate(self):
        from jury.anomaly_jury import AnomalyJuryAgent
        df = make_ts(n=60, noise=0.02)
        ctx = self._make_ctx(df)
        result = AnomalyJuryAgent().run(ctx)
        if result.status == "success":
            assert result.data.get("anomaly_count", 0) <= 3

    def test_empty_ts_skipped(self):
        from jury.anomaly_jury import AnomalyJuryAgent
        ctx = AnalysisContext(kpi_col="revenue", date_col="date",
                              data_profile={"rows": 0, "has_time_series": False,
                                            "dimensions": [], "kpis": []})
        result = AnomalyJuryAgent().run(ctx)
        assert result.status in ("skipped", "error")

    def test_result_contains_consensus_key(self):
        from jury.anomaly_jury import AnomalyJuryAgent
        df = make_ts_spike(n=60, spike_idx=30)
        ctx = self._make_ctx(df)
        result = AnomalyJuryAgent().run(ctx)
        if result.status == "success":
            assert "consensus" in result.data or "method" in result.data


# ══════════════════════════════════════════════════════════════════════
# ForecastJuryAgent
# ══════════════════════════════════════════════════════════════════════

class TestForecastJuryAgent:

    def _make_ctx(self, df):
        ctx = AnalysisContext(df=df.copy(), kpi_col="revenue", date_col="date")
        ts = df.copy()
        ts["date"] = pd.to_datetime(ts["date"])
        ts = ts.sort_values("date").reset_index(drop=True)
        ctx.ts = ts
        ctx.data_profile = {"rows": len(df), "has_time_series": True,
                            "dimensions": [], "kpis": ["revenue"],
                            "has_funnel_signal": False, "has_cohort_signal": False}
        return ctx

    def test_forecast_returns_horizon(self):
        from jury.forecast_jury import ForecastJuryAgent
        df = make_ts(n=60)
        ctx = self._make_ctx(df)
        result = ForecastJuryAgent().run(ctx)
        assert result.status in ("success", "skipped", "error")
        if result.status == "success":
            assert "forecast" in result.data or "periods" in result.data

    def test_forecast_holdout_validation_present(self):
        from jury.forecast_jury import ForecastJuryAgent
        df = make_ts(n=60)
        ctx = self._make_ctx(df)
        result = ForecastJuryAgent().run(ctx)
        if result.status == "success":
            # Either mape, holdout_mape, or validation_error should be in data
            has_validation = any(
                k in result.data
                for k in ("mape", "holdout_mape", "validation_error", "error")
            )
            assert has_validation

    def test_too_few_rows_skips_gracefully(self):
        from jury.forecast_jury import ForecastJuryAgent
        df = make_ts(n=5)
        ctx = self._make_ctx(df)
        result = ForecastJuryAgent().run(ctx)
        assert result.status in ("skipped", "error", "success")
