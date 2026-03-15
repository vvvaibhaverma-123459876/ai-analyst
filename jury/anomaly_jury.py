"""
jury/anomaly_jury.py
Anomaly jury — 4 jurors with independent methods.
Requires 3/4 agreement before promoting a finding.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from jury.base_juror import BaseJuror, JurorVerdict
from jury.foreman import Foreman
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from analysis.anomaly_detector import AnomalyDetector
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)


class ZScoreJuror(BaseJuror):
    name = "zscore_juror"; method = "z-score"

    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        ts, date_col, kpi_col = context.ts, context.date_col, context.kpi_col
        if ts.empty or kpi_col not in ts.columns:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — no time series", "skipped")
        detector = AnomalyDetector(window=config.DEFAULT_ROLLING_WINDOW,
                                   z_threshold=config.DEFAULT_Z_THRESHOLD)
        result = detector.detect_zscore(ts, date_col, kpi_col)
        n = int(result["anomaly"].sum())
        records = detector.summarise(result, date_col, kpi_col)
        conf = min(0.9, 0.5 + n * 0.1) if n > 0 else 0.1
        return JurorVerdict(self.name, self.method,
                            {"anomaly_count": n, "method": "z-score", "records": records},
                            conf, f"Z-score: {n} anomalies detected", "success")


class IQRJuror(BaseJuror):
    name = "iqr_juror"; method = "iqr"

    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        ts, kpi_col = context.ts, context.kpi_col
        if ts.empty or kpi_col not in ts.columns:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — no time series", "skipped")
        detector = AnomalyDetector()
        result = detector.detect_iqr(ts, kpi_col)
        n = int(result["anomaly"].sum())
        conf = min(0.85, 0.5 + n * 0.1) if n > 0 else 0.1
        return JurorVerdict(self.name, self.method,
                            {"anomaly_count": n, "method": "iqr"},
                            conf, f"IQR: {n} anomalies detected", "success")


class STLJuror(BaseJuror):
    name = "stl_juror"; method = "stl"

    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        ts, date_col, kpi_col = context.ts, context.date_col, context.kpi_col
        if ts.empty or len(ts) < 28 or kpi_col not in ts.columns:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — need ≥28 points for STL", "skipped")
        detector = AnomalyDetector()
        result = detector.detect_stl(ts, date_col, kpi_col)
        n = int(result["anomaly"].sum())
        conf = min(0.92, 0.55 + n * 0.1) if n > 0 else 0.1
        return JurorVerdict(self.name, self.method,
                            {"anomaly_count": n, "method": "stl"},
                            conf, f"STL: {n} anomalies detected", "success")


class IsolationForestJuror(BaseJuror):
    name = "iforest_juror"; method = "isolation_forest"

    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        df = context.df
        kpi_col = context.kpi_col
        if df.empty or len(df) < 20:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — need ≥20 rows", "skipped")
        try:
            from sklearn.ensemble import IsolationForest
            numeric = df.select_dtypes(include=[np.number])
            if numeric.empty:
                return JurorVerdict(self.name, self.method, {}, 0.0,
                                    "Skipped — no numeric columns", "skipped")
            X = numeric.fillna(numeric.mean())
            clf = IsolationForest(contamination=0.1, random_state=42)
            labels = clf.fit_predict(X)
            n = int((labels == -1).sum())
            conf = min(0.88, 0.5 + n * 0.08) if n > 0 else 0.1
            return JurorVerdict(self.name, self.method,
                                {"anomaly_count": n, "method": "isolation_forest",
                                 "multivariate": True},
                                conf, f"Isolation Forest: {n} multivariate anomalies", "success")
        except ImportError:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — scikit-learn not installed", "skipped")
        except Exception as e:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                f"Error: {e}", "error", str(e))


class AnomalyJuryAgent(BaseAgent):
    name = "anomaly"
    description = "Anomaly jury: Z-score + IQR + STL + Isolation Forest"

    def _run(self, context: AnalysisContext) -> AgentResult:
        if context.ts.empty:
            return self.skip("No time series in context.")

        jurors = [ZScoreJuror(), IQRJuror(), STLJuror(), IsolationForestJuror()]
        foreman = Foreman("anomaly", jurors)
        fv = foreman.deliberate(context)

        # Write enriched ts back if available
        if fv.primary_finding.get("ts_with_anomalies") is not None:
            context.ts = fv.primary_finding["ts_with_anomalies"]

        result = foreman.to_agent_result(self.name, fv)

        # Populate expected fields for downstream compatibility
        if "anomaly_count" not in result.data:
            result.data["anomaly_count"] = fv.primary_finding.get("anomaly_count", 0)
        result.data["method_used"] = f"jury/{fv.consensus}"
        result.data["severity_counts"] = {"high": 0, "medium": 0, "low": 0}
        result.data["anomaly_records"] = fv.primary_finding.get("records", [])

        return result
