"""
analysis/anomaly_detector.py
Anomaly detection: z-score (from app.py), IQR, and STL decomposition.
"""

import pandas as pd
import numpy as np
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:

    def __init__(
        self,
        window: int = None,
        z_threshold: float = None,
    ):
        self.window = window or config.DEFAULT_ROLLING_WINDOW
        self.z_threshold = z_threshold or config.DEFAULT_Z_THRESHOLD

    # ------------------------------------------------------------------
    # Z-score method (preserved from app.py)
    # ------------------------------------------------------------------

    def detect_zscore(
        self,
        ts: pd.DataFrame,
        date_col: str,
        value_col: str,
        window: int = None,
        z_threshold: float = None,
    ) -> pd.DataFrame:
        """
        Rolling z-score anomaly detection.
        Preserved and extracted from app.py v0.1.
        Returns ts with added columns: mean, std, zscore, anomaly, severity.
        """
        w = window or self.window
        zt = z_threshold or self.z_threshold

        result = ts.copy()
        result["mean"] = result[value_col].rolling(w).mean()
        result["std"] = result[value_col].rolling(w).std()
        result["zscore"] = (result[value_col] - result["mean"]) / result["std"]
        result["anomaly"] = result["zscore"].abs() > zt
        result["severity"] = result["zscore"].abs().apply(
            lambda z: "high" if z > 3 else ("medium" if z > 2 else "low")
            if not np.isnan(z) else "unknown"
        )
        n = result["anomaly"].sum()
        logger.info(f"Z-score anomaly detection: {n} anomalies detected (window={w}, threshold={zt})")
        return result

    # ------------------------------------------------------------------
    # IQR method
    # ------------------------------------------------------------------

    def detect_iqr(
        self,
        ts: pd.DataFrame,
        value_col: str,
        multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """
        IQR-based outlier detection. Good for skewed distributions.
        """
        result = ts.copy()
        q1 = result[value_col].quantile(0.25)
        q3 = result[value_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        result["iqr_lower"] = lower
        result["iqr_upper"] = upper
        result["anomaly"] = (result[value_col] < lower) | (result[value_col] > upper)
        n = result["anomaly"].sum()
        logger.info(f"IQR anomaly detection: {n} anomalies (Q1={q1:.2f}, Q3={q3:.2f})")
        return result

    # ------------------------------------------------------------------
    # STL decomposition (requires statsmodels)
    # ------------------------------------------------------------------

    def detect_stl(
        self,
        ts: pd.DataFrame,
        date_col: str,
        value_col: str,
        period: int = 7,
        residual_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        STL decomposition-based anomaly detection.
        Flags points where the residual component is an outlier.
        Requires: statsmodels
        """
        try:
            from statsmodels.tsa.seasonal import STL
        except ImportError:
            logger.warning("statsmodels not installed. Falling back to z-score.")
            return self.detect_zscore(ts, date_col, value_col)

        result = ts.copy().set_index(date_col)
        series = result[value_col].dropna()

        if len(series) < period * 2:
            logger.warning("Not enough data for STL. Falling back to z-score.")
            return self.detect_zscore(ts, date_col, value_col)

        stl = STL(series, period=period, robust=True)
        fit = stl.fit()

        residuals = fit.resid
        res_mean = residuals.mean()
        res_std = residuals.std()
        z_resid = (residuals - res_mean) / res_std

        result = result.reset_index()
        result["stl_trend"] = fit.trend.values
        result["stl_seasonal"] = fit.seasonal.values
        result["stl_residual"] = fit.resid.values
        result["anomaly"] = z_resid.abs().values > residual_threshold
        result["severity"] = z_resid.abs().apply(
            lambda z: "high" if z > 3 else "medium" if z > 2 else "low"
        ).values

        n = result["anomaly"].sum()
        logger.info(f"STL anomaly detection: {n} anomalies (period={period})")
        return result

    # ------------------------------------------------------------------
    # Summary helper
    # ------------------------------------------------------------------

    def summarise(self, ts: pd.DataFrame, date_col: str, value_col: str) -> list[dict]:
        """Returns list of anomaly dicts for use in LLM payloads."""
        anomalies = ts[ts["anomaly"] == True]
        return anomalies[[date_col, value_col]].rename(
            columns={date_col: "date", value_col: "value"}
        ).to_dict("records")
