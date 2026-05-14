"""
analysis/anomaly_detector.py
Robust anomaly detection for product analytics time series.

Capabilities:
- Canonical AnalysisContract interface via analyze()/detect()
- Robust baseline detection using trailing rolling median + MAD
- Z-score, IQR, and STL helper methods for backward compatibility
- Severity, baseline, percent-delta, and direction per anomaly
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.contract import AnalysisContract, AnalysisResult
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector(AnalysisContract):
    module_name = "anomaly"

    def __init__(self, window: int = None, z_threshold: float = None):
        self.window = int(window or config.DEFAULT_ROLLING_WINDOW)
        self.z_threshold = float(z_threshold or config.DEFAULT_Z_THRESHOLD)

    # ------------------------------------------------------------------
    # Contract entry point
    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame, **kwargs) -> AnalysisResult:
        kpi_col = kwargs.get("kpi_col") or kwargs.get("value_col", "")
        date_col = kwargs.get("date_col", "")
        method = kwargs.get("method", "auto")
        errors = self.validate_inputs(df, required_cols=[kpi_col] if kpi_col else [], min_rows=3)
        if errors:
            return AnalysisResult(module=self.module_name, ok=False, errors=errors, method=method)
        return self.detect(df, kpi_col=kpi_col, date_col=date_col, method=method)

    def detect(
        self,
        df: pd.DataFrame,
        kpi_col: str = "",
        date_col: str = "",
        method: str = "auto",
        **kwargs,
    ) -> AnalysisResult:
        if not kpi_col and "value_col" in kwargs:
            kpi_col = kwargs["value_col"]
        if not kpi_col or kpi_col not in df.columns:
            return AnalysisResult(
                module=self.module_name,
                ok=False,
                errors=[f"Column '{kpi_col}' not found in DataFrame."],
                method=method,
            )

        errors = self.validate_inputs(df, required_cols=[kpi_col], min_rows=3)
        if errors:
            return AnalysisResult(module=self.module_name, ok=False, errors=errors, method=method)

        ordered = self._ordered(df, date_col)
        s = pd.to_numeric(ordered[kpi_col], errors="coerce")
        if s.dropna().empty or float(s.dropna().std() or 0) == 0:
            return AnalysisResult(
                module=self.module_name,
                ok=True,
                anomaly_count=0,
                method=method,
                summary="No anomalies — series is constant or empty.",
                records=[],
            )

        method_l = (method or "auto").lower()
        if method_l == "iqr":
            annotated = self.detect_iqr(ordered, kpi_col=kpi_col)
            used = "iqr"
        elif method_l == "stl" and date_col:
            annotated = self.detect_stl(ordered, date_col, kpi_col)
            used = "stl"
        elif method_l == "zscore":
            annotated = self.detect_zscore(ordered, date_col, kpi_col)
            used = "zscore"
        else:
            annotated = self.detect_robust(ordered, date_col, kpi_col)
            used = "robust"

        records = self.summarise(annotated, date_col, kpi_col) if date_col in annotated.columns else self._records_no_date(annotated, kpi_col)
        n = len(records)
        high = sum(1 for r in records if r.get("severity") == "high")
        summary = (
            "No anomalies detected."
            if n == 0
            else f"{n} anomaly point(s) detected via {used}; high severity={high}."
        )
        return AnalysisResult(
            module=self.module_name,
            ok=True,
            anomaly_count=n,
            records=records[:20],
            summary=summary,
            confidence=min(0.95, 0.55 + n * 0.10),
            method=used if used != "robust" else "zscore",
            metadata={
                "window": self.window,
                "z_threshold": self.z_threshold,
                "method_used": used,
                "anomaly_records": records[:20],
                "annotated_columns": [c for c in annotated.columns if c not in df.columns],
            },
        )

    # ------------------------------------------------------------------
    # Detection implementations
    # ------------------------------------------------------------------
    def _ordered(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        out = df.copy()
        if date_col and date_col in out.columns:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
            out = out.sort_values(date_col).reset_index(drop=True)
        return out

    def _zscore_series(self, df: pd.DataFrame, col: str) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce").ffill().bfill()
        # Use a trailing baseline that excludes the current row so a spike cannot
        # inflate its own mean/std and hide itself.
        hist = s.shift(1)
        mean = hist.rolling(self.window, min_periods=max(3, min(self.window, 5))).mean()
        std = hist.rolling(self.window, min_periods=max(3, min(self.window, 5))).std(ddof=0)
        global_mean = hist.expanding(min_periods=3).mean()
        global_std = hist.expanding(min_periods=3).std(ddof=0)
        mean = mean.fillna(global_mean).fillna(s.mean())
        std = std.replace(0, np.nan).fillna(global_std).replace(0, np.nan).fillna(float(s.std(ddof=0) or 1.0))
        return (s - mean) / std.replace(0, 1.0)

    def _zscore_flags(self, df: pd.DataFrame, col: str) -> pd.Series:
        return self._zscore_series(df, col).abs() >= self.z_threshold

    def _iqr_flags(self, df: pd.DataFrame, col: str, multiplier: float = 1.5) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            return pd.Series(False, index=df.index)
        return (s < q1 - multiplier * iqr) | (s > q3 + multiplier * iqr)

    def _stl_flags(self, df: pd.DataFrame, date_col: str, col: str, period: int = 7, threshold: float = 2.5) -> pd.Series:
        try:
            from statsmodels.tsa.seasonal import STL
            s = pd.to_numeric(df[col], errors="coerce").interpolate().ffill().bfill()
            if len(s) < period * 2:
                return self._zscore_flags(df, col)
            fit = STL(s, period=period, robust=True).fit()
            resid = pd.Series(fit.resid, index=df.index)
            med = resid.rolling(self.window, min_periods=3).median()
            mad = (resid - med).abs().rolling(self.window, min_periods=3).median().replace(0, np.nan)
            robust_z = 0.6745 * (resid - med) / mad
            return robust_z.abs().fillna(0) >= threshold
        except Exception:
            return self._zscore_flags(df, col)

    def detect_robust(self, ts: pd.DataFrame, date_col: str, value_col: str = "", *, kpi_col: str = "") -> pd.DataFrame:
        col = kpi_col or value_col
        result = self._ordered(ts, date_col)
        s = pd.to_numeric(result[col], errors="coerce").ffill().bfill()
        hist = s.shift(1)
        min_periods = max(3, min(self.window, 5))
        baseline = hist.rolling(self.window, min_periods=min_periods).median()
        abs_dev = (hist - baseline).abs()
        mad = abs_dev.rolling(self.window, min_periods=min_periods).median()
        exp_median = hist.expanding(min_periods=3).median()
        exp_mad = (hist - exp_median).abs().expanding(min_periods=3).median()
        baseline = baseline.fillna(exp_median).fillna(s.median())
        mad = mad.replace(0, np.nan).fillna(exp_mad).replace(0, np.nan)
        fallback_scale = float(np.nanmedian(np.abs(s - np.nanmedian(s))) or s.std(ddof=0) or 1.0)
        mad = mad.fillna(fallback_scale).replace(0, 1.0)
        robust_z = 0.6745 * (s - baseline) / mad
        pct_delta = (s - baseline) / baseline.replace(0, np.nan) * 100
        result["baseline"] = baseline
        result["zscore"] = robust_z.replace([np.inf, -np.inf], np.nan)
        result["pct_delta_from_baseline"] = pct_delta.replace([np.inf, -np.inf], np.nan)
        result["direction"] = np.where(s >= baseline, "spike", "drop")
        result["anomaly"] = result["zscore"].abs().fillna(0) >= self.z_threshold
        result["severity"] = result["zscore"].abs().apply(self._severity)
        return result

    # Legacy methods returning annotated DataFrame
    def detect_zscore(self, ts: pd.DataFrame, date_col: str, value_col: str = "", window: int = None, z_threshold: float = None, *, kpi_col: str = "") -> pd.DataFrame:
        col = kpi_col or value_col
        old_w, old_z = self.window, self.z_threshold
        if window is not None:
            self.window = int(window)
        if z_threshold is not None:
            self.z_threshold = float(z_threshold)
        try:
            result = self.detect_robust(ts, date_col, col)
            return result
        finally:
            self.window, self.z_threshold = old_w, old_z

    def detect_iqr(self, ts: pd.DataFrame, value_col: str = "", kpi_col: str = "", multiplier: float = 1.5) -> pd.DataFrame:
        col = kpi_col or value_col
        result = ts.copy()
        s = pd.to_numeric(result[col], errors="coerce")
        flags = self._iqr_flags(result, col, multiplier)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        result["iqr_lower"] = q1 - multiplier * iqr
        result["iqr_upper"] = q3 + multiplier * iqr
        result["anomaly"] = flags
        result["zscore"] = self._zscore_series(result, col)
        result["baseline"] = s.median()
        result["direction"] = np.where(s >= result["baseline"], "spike", "drop")
        result["severity"] = result["zscore"].abs().apply(self._severity)
        return result

    def detect_stl(self, ts: pd.DataFrame, date_col: str, value_col: str = "", *, kpi_col: str = "") -> pd.DataFrame:
        col = kpi_col or value_col
        result = self._ordered(ts, date_col)
        result["anomaly"] = self._stl_flags(result, date_col, col)
        result["zscore"] = self._zscore_series(result, col)
        s = pd.to_numeric(result[col], errors="coerce")
        result["baseline"] = s.rolling(self.window, min_periods=3).median().shift(1).fillna(s.median())
        result["direction"] = np.where(s >= result["baseline"], "spike", "drop")
        result["severity"] = result["zscore"].abs().apply(self._severity)
        return result

    def _severity(self, z) -> str:
        try:
            z = abs(float(z))
        except Exception:
            return "unknown"
        if z >= max(4.0, self.z_threshold + 1.5):
            return "high"
        if z >= self.z_threshold:
            return "medium"
        return "low"

    def _records_no_date(self, ts: pd.DataFrame, value_col: str) -> list[dict]:
        rows = ts[ts.get("anomaly", False) == True]
        records = []
        for idx, row in rows.iterrows():
            records.append({
                "index": int(idx),
                "value": float(pd.to_numeric(row[value_col], errors="coerce")),
                "z_score": round(float(row.get("zscore", np.nan)), 2) if pd.notna(row.get("zscore", np.nan)) else None,
                "severity": row.get("severity", "unknown"),
                "direction": row.get("direction", "unknown"),
            })
        return records

    def summarise(self, ts: pd.DataFrame, date_col: str, value_col: str) -> list[dict]:
        if "anomaly" not in ts.columns:
            return []
        rows = ts[ts["anomaly"] == True]
        records: list[dict] = []
        for _, row in rows.iterrows():
            value = pd.to_numeric(row.get(value_col), errors="coerce")
            rec = {
                "date": str(row.get(date_col, ""))[:10],
                "value": float(value) if pd.notna(value) else None,
            }
            if "zscore" in row:
                rec["z_score"] = round(float(row["zscore"]), 2) if pd.notna(row["zscore"]) else None
            if "severity" in row:
                rec["severity"] = row["severity"]
            if "direction" in row:
                rec["direction"] = row["direction"]
            if "baseline" in row and pd.notna(row["baseline"]):
                rec["baseline"] = float(row["baseline"])
            if "pct_delta_from_baseline" in row and pd.notna(row["pct_delta_from_baseline"]):
                rec["pct_delta_from_baseline"] = round(float(row["pct_delta_from_baseline"]), 2)
            records.append(rec)
        return records
