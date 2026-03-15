"""
analysis/anomaly_detector.py  — v9
Anomaly detection: z-score, IQR, STL decomposition.

v9 changes
----------
- Implements AnalysisContract (analyze / validate_inputs / to_benchmark_output)
- detect() is the canonical entry point, accepting kpi_col (alias: value_col)
- detect_zscore / detect_iqr / detect_stl preserved as internal helpers
- Returns AnalysisResult with .to_agent_data() for backward-compat AgentResult.data
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from analysis.contract import AnalysisContract, AnalysisResult
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector(AnalysisContract):

    module_name = "anomaly"

    def __init__(self, window: int = None, z_threshold: float = None):
        self.window      = window      or config.DEFAULT_ROLLING_WINDOW
        self.z_threshold = z_threshold or config.DEFAULT_Z_THRESHOLD

    # ------------------------------------------------------------------
    # AnalysisContract: canonical entry point
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, **kwargs) -> AnalysisResult:
        """
        Canonical contract method.  Accepts kpi_col or value_col (alias).
        Returns AnalysisResult — use .to_agent_data() for AgentResult.data.
        """
        kpi_col  = kwargs.get("kpi_col") or kwargs.get("value_col", "")
        date_col = kwargs.get("date_col", "")
        method   = kwargs.get("method", "auto")

        errors = self.validate_inputs(df, required_cols=[kpi_col] if kpi_col else [], min_rows=3)
        if errors:
            return AnalysisResult(module=self.module_name, ok=False,
                                  errors=errors, method=method)

        return self.detect(df, kpi_col=kpi_col, date_col=date_col, method=method)

    # ------------------------------------------------------------------
    # detect() — primary public method (contract-aligned)
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame, kpi_col: str = "", date_col: str = "",
               method: str = "auto", **_) -> AnalysisResult:
        """
        Entry point used by agents and benchmarks.
        Accepts kpi_col; value_col is an accepted alias (set via **_).
        method: "auto" | "zscore" | "iqr" | "stl"
        Returns AnalysisResult.
        """
        # Accept value_col as alias for kpi_col
        if not kpi_col and "value_col" in _:
            kpi_col = _["value_col"]

        if not kpi_col or kpi_col not in df.columns:
            return AnalysisResult(module=self.module_name, ok=False,
                                  errors=[f"Column '{kpi_col}' not found in DataFrame."],
                                  method=method)

        errors = self.validate_inputs(df, required_cols=[kpi_col], min_rows=3)
        if errors:
            return AnalysisResult(module=self.module_name, ok=False,
                                  errors=errors, method=method)

        series = pd.to_numeric(df[kpi_col], errors="coerce")
        if series.dropna().empty or series.std() == 0:
            return AnalysisResult(module=self.module_name, ok=True,
                                  anomaly_count=0, method=method,
                                  summary="No anomalies — series is constant or empty.",
                                  records=[])

        if method == "iqr":
            flagged = self._iqr_flags(df, kpi_col)
            used = "iqr"
        elif method == "stl" and date_col:
            flagged = self._stl_flags(df, date_col, kpi_col)
            used = "stl"
        else:
            flagged = self._zscore_flags(df, kpi_col)
            used = "zscore"

        # Build records
        records = []
        if date_col and date_col in df.columns:
            for i, row in df[flagged].iterrows():
                records.append({
                    "date":    str(row[date_col])[:10],
                    "value":   float(pd.to_numeric(row[kpi_col], errors="coerce")),
                    "z_score": round(float(self._zscore_series(df, kpi_col).iloc[i]), 2)
                               if used == "zscore" else None,
                })
        else:
            records = [{"index": int(i), "value": float(pd.to_numeric(df[kpi_col].iloc[i], errors="coerce"))}
                       for i in df.index[flagged]]

        n = int(flagged.sum())
        return AnalysisResult(
            module=self.module_name, ok=True,
            anomaly_count=n,
            records=records[:20],
            summary=f"{n} anomaly point(s) detected via {used}.",
            confidence=min(0.9, 0.5 + n * 0.08),
            method=used,
            metadata={"window": self.window, "z_threshold": self.z_threshold},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _zscore_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce").fillna(method="ffill").fillna(0)
        mu  = s.rolling(self.window, min_periods=1).mean()
        std = s.rolling(self.window, min_periods=1).std().replace(0, 1)
        return (s - mu) / std

    def _zscore_flags(self, df: pd.DataFrame, col: str) -> pd.Series:
        return self._zscore_series(df, col).abs() > self.z_threshold

    def _iqr_flags(self, df: pd.DataFrame, col: str,
                   multiplier: float = 1.5) -> pd.Series:
        s  = pd.to_numeric(df[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - multiplier * iqr) | (s > q3 + multiplier * iqr)

    def _stl_flags(self, df: pd.DataFrame, date_col: str, col: str,
                   period: int = 7, threshold: float = 2.0) -> pd.Series:
        try:
            from statsmodels.tsa.seasonal import STL
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) < period * 2:
                return self._zscore_flags(df, col)
            fit = STL(s, period=period, robust=True).fit()
            resid = fit.resid
            z = (resid - resid.mean()) / (resid.std() or 1)
            flags = pd.Series(False, index=df.index)
            flags.iloc[:len(z)] = z.abs().values > threshold
            return flags
        except ImportError:
            return self._zscore_flags(df, col)

    # ------------------------------------------------------------------
    # Legacy interface (preserved for backward compat — internally delegate)
    # ------------------------------------------------------------------

    def detect_zscore(self, ts: pd.DataFrame, date_col: str,
                      value_col: str = "", kpi_col: str = "",
                      window: int = None, z_threshold: float = None) -> pd.DataFrame:
        """Legacy method — returns annotated DataFrame as before."""
        col = kpi_col or value_col
        w   = window or self.window
        zt  = z_threshold or self.z_threshold
        result = ts.copy()
        s = pd.to_numeric(result[col], errors="coerce")
        result["mean"]    = s.rolling(w).mean()
        result["std"]     = s.rolling(w).std()
        result["zscore"]  = (s - result["mean"]) / result["std"].replace(0, 1)
        result["anomaly"] = result["zscore"].abs() > zt
        result["severity"] = result["zscore"].abs().apply(
            lambda z: "high" if z > 3 else ("medium" if z > 2 else "low")
            if not np.isnan(z) else "unknown"
        )
        return result

    def detect_iqr(self, ts: pd.DataFrame, value_col: str = "",
                   kpi_col: str = "", multiplier: float = 1.5) -> pd.DataFrame:
        """Legacy method — returns annotated DataFrame."""
        col = kpi_col or value_col
        result = ts.copy()
        s  = pd.to_numeric(result[col], errors="coerce")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        result["iqr_lower"] = q1 - multiplier * iqr
        result["iqr_upper"] = q3 + multiplier * iqr
        result["anomaly"]   = (s < result["iqr_lower"]) | (s > result["iqr_upper"])
        return result

    def summarise(self, ts: pd.DataFrame, date_col: str, value_col: str) -> list[dict]:
        anomalies = ts[ts["anomaly"] == True]
        return anomalies[[date_col, value_col]].rename(
            columns={date_col: "date", value_col: "value"}
        ).to_dict("records")
