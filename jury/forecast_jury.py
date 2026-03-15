"""
jury/forecast_jury.py
Forecast jury — Prophet + ARIMA + ETS + holdout-validated ensemble.
Guardian enforces temporal holdout before any forecast is published.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from jury.base_juror import BaseJuror, JurorVerdict
from jury.foreman import Foreman
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.logger import get_logger

logger = get_logger(__name__)

HOLDOUT_FRACTION = 0.20   # last 20% always reserved for validation


def _compute_holdout_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """MAPE — lower is better."""
    mask = actual != 0
    if not mask.any():
        return 1.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


class ProphetJuror(BaseJuror):
    name = "prophet_juror"; method = "prophet"

    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        ts, date_col, kpi_col = context.ts, context.date_col, context.kpi_col
        if ts.empty or len(ts) < 60:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — need ≥60 points", "skipped")
        try:
            from prophet import Prophet
            n = len(ts)
            split = int(n * (1 - HOLDOUT_FRACTION))
            train = ts.iloc[:split][[date_col, kpi_col]].rename(
                columns={date_col: "ds", kpi_col: "y"})
            test_actual = ts[kpi_col].iloc[split:].values

            m = Prophet(interval_width=0.80, changepoint_prior_scale=0.05)
            m.fit(train)
            future = m.make_future_dataframe(periods=len(test_actual))
            fcst = m.predict(future)
            test_pred = fcst["yhat"].iloc[split:].values

            mape = _compute_holdout_error(test_actual, test_pred)
            conf = max(0.1, min(0.95, 1.0 - mape))

            horizon = max(7, n // 4)
            full_future = m.make_future_dataframe(periods=horizon)
            full_fcst = m.predict(full_future)
            fdf = full_fcst[full_fcst["ds"] > ts[date_col].max()][
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].reset_index(drop=True)

            last_actual = float(ts[kpi_col].iloc[-1])
            last_forecast = float(fdf["yhat"].iloc[-1]) if not fdf.empty else last_actual
            direction = "up" if last_forecast > last_actual else "down"
            pct = abs((last_forecast - last_actual) / last_actual * 100) if last_actual else 0

            return JurorVerdict(self.name, self.method, {
                "forecast_df": fdf, "method": "Prophet",
                "direction": direction, "pct_change": round(pct, 2),
                "last_actual": round(last_actual, 2),
                "last_forecast": round(last_forecast, 2),
                "holdout_mape": round(mape, 3), "horizon": horizon,
            }, conf, f"Prophet: {direction} {pct:.1f}% (MAPE={mape:.2f})", "success")
        except ImportError:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — prophet not installed", "skipped")
        except Exception as e:
            return JurorVerdict(self.name, self.method, {}, 0.0, f"Error: {e}", "error", str(e))


class ARIMAJuror(BaseJuror):
    name = "arima_juror"; method = "arima"

    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        ts, date_col, kpi_col = context.ts, context.date_col, context.kpi_col
        if ts.empty or len(ts) < 30:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — need ≥30 points", "skipped")
        try:
            from statsmodels.tsa.arima.model import ARIMA
            series = ts[kpi_col].dropna().values.astype(float)
            n = len(series)
            split = int(n * (1 - HOLDOUT_FRACTION))

            model = ARIMA(series[:split], order=(1, 1, 1))
            fit = model.fit()
            test_pred = fit.forecast(steps=n - split)
            test_actual = series[split:]
            mape = _compute_holdout_error(test_actual, test_pred)
            conf = max(0.1, min(0.90, 1.0 - mape))

            horizon = max(7, n // 4)
            full_model = ARIMA(series, order=(1, 1, 1)).fit()
            fcst_result = full_model.get_forecast(steps=horizon)
            mean = fcst_result.predicted_mean
            ci = fcst_result.conf_int(alpha=0.20)

            last_date = pd.to_datetime(ts[date_col].iloc[-1])
            future_dates = pd.date_range(last_date, periods=horizon + 1, freq="D")[1:]
            fdf = pd.DataFrame({
                "ds": future_dates, "yhat": mean,
                "yhat_lower": ci.iloc[:, 0].values,
                "yhat_upper": ci.iloc[:, 1].values,
            })

            last_actual = float(series[-1])
            last_forecast = float(mean[-1])
            direction = "up" if last_forecast > last_actual else "down"
            pct = abs((last_forecast - last_actual) / last_actual * 100) if last_actual else 0

            return JurorVerdict(self.name, self.method, {
                "forecast_df": fdf, "method": "ARIMA",
                "direction": direction, "pct_change": round(pct, 2),
                "last_actual": round(last_actual, 2),
                "last_forecast": round(last_forecast, 2),
                "holdout_mape": round(mape, 3), "horizon": horizon,
            }, conf, f"ARIMA: {direction} {pct:.1f}% (MAPE={mape:.2f})", "success")
        except Exception as e:
            return JurorVerdict(self.name, self.method, {}, 0.0, f"Error: {e}", "error", str(e))


class ETSJuror(BaseJuror):
    name = "ets_juror"; method = "holt_ets"

    def deliberate(self, context: AnalysisContext) -> JurorVerdict:
        ts, date_col, kpi_col = context.ts, context.date_col, context.kpi_col
        if ts.empty or len(ts) < 10:
            return JurorVerdict(self.name, self.method, {}, 0.0,
                                "Skipped — need ≥10 points", "skipped")

        series = ts[kpi_col].dropna().values.astype(float)
        n = len(series)
        split = int(n * (1 - HOLDOUT_FRACTION))

        # Holt's linear trend
        alpha, beta = 0.3, 0.1
        level, trend = [series[0]], [series[1] - series[0]]
        for val in series[1:]:
            l = alpha * val + (1 - alpha) * (level[-1] + trend[-1])
            t = beta * (l - level[-1]) + (1 - beta) * trend[-1]
            level.append(l); trend.append(t)

        test_pred = np.array([level[split-1] + i * trend[split-1]
                               for i in range(1, n - split + 1)])
        test_actual = series[split:]
        mape = _compute_holdout_error(test_actual, test_pred)
        conf = max(0.1, min(0.75, 1.0 - mape))   # ETS capped at 0.75

        horizon = max(7, n // 4)
        forecasts = [level[-1] + i * trend[-1] for i in range(1, horizon + 1)]

        last_date = pd.to_datetime(ts[date_col].iloc[-1])
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq="D")[1:]
        fdf = pd.DataFrame({
            "ds": future_dates, "yhat": forecasts,
            "yhat_lower": [f * 0.9 for f in forecasts],
            "yhat_upper": [f * 1.1 for f in forecasts],
        })

        last_actual = float(series[-1])
        last_forecast = float(forecasts[-1])
        direction = "up" if last_forecast > last_actual else "down"
        pct = abs((last_forecast - last_actual) / last_actual * 100) if last_actual else 0

        return JurorVerdict(self.name, self.method, {
            "forecast_df": fdf, "method": "Holt ETS",
            "direction": direction, "pct_change": round(pct, 2),
            "last_actual": round(last_actual, 2),
            "last_forecast": round(last_forecast, 2),
            "holdout_mape": round(mape, 3), "horizon": horizon,
        }, conf, f"ETS: {direction} {pct:.1f}% (MAPE={mape:.2f})", "success")


class ForecastJuryAgent(BaseAgent):
    name = "forecast"
    description = "Forecast jury: Prophet + ARIMA + ETS with temporal holdout validation"

    def _run(self, context: AnalysisContext) -> AgentResult:
        if context.ts.empty:
            return self.skip("No time series.")
        if len(context.ts) < 10:
            return self.skip(f"Only {len(context.ts)} points — need ≥10.")

        jurors = [ProphetJuror(), ARIMAJuror(), ETSJuror()]
        foreman = Foreman("forecast", jurors)
        fv = foreman.deliberate(context)
        result = foreman.to_agent_result(self.name, fv)

        # Ensure expected fields exist
        pf = fv.primary_finding
        for key in ["forecast_df", "method", "direction", "pct_change",
                    "last_actual", "last_forecast", "horizon"]:
            if key not in result.data:
                result.data[key] = pf.get(key)

        # Best MAPE across jurors
        mapes = [
            v.finding.get("holdout_mape")
            for v in fv.all_verdicts
            if v.status == "success" and v.finding.get("holdout_mape") is not None
        ]
        if mapes:
            result.data["best_holdout_mape"] = round(min(mapes), 3)

        return result
