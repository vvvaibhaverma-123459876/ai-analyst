"""
agents/forecast_agent.py
Forecast Agent — forward-looking time series prediction.

Methods (auto-selected by data length):
  - Prophet  (≥ 60 data points, handles seasonality + holidays)
  - ARIMA    (30–59 points)
  - ETS/Holt (< 30 points, simple exponential smoothing)

Always returns:
  - forecast_df: DataFrame with ds, yhat, yhat_lower, yhat_upper
  - horizon: number of periods forecast
  - method: which model was used
  - summary: plain-text trend + prediction statement
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_HORIZON = 14   # periods forward


class ForecastAgent(BaseAgent):
    name = "forecast"
    description = "Forward-looking prediction: Prophet / ARIMA / ETS auto-selected by data length"

    def _run(self, context: AnalysisContext) -> AgentResult:
        ts = context.ts
        date_col = context.date_col
        kpi_col = context.kpi_col

        if ts.empty or not kpi_col or kpi_col not in ts.columns:
            return self.skip("No time series available — run Trend agent first.")

        n = len(ts)
        if n < 10:
            return self.skip(f"Only {n} data points — need at least 10 to forecast.")

        horizon = min(DEFAULT_HORIZON, max(7, n // 4))

        if n >= 60:
            return self._prophet(ts, date_col, kpi_col, horizon)
        elif n >= 30:
            return self._arima(ts, date_col, kpi_col, horizon)
        else:
            return self._ets(ts, date_col, kpi_col, horizon)

    # ------------------------------------------------------------------
    # Prophet
    # ------------------------------------------------------------------
    def _prophet(self, ts, date_col, kpi_col, horizon) -> AgentResult:
        try:
            from prophet import Prophet
        except ImportError:
            self.logger.warning("prophet not installed — falling back to ARIMA")
            return self._arima(ts, date_col, kpi_col, horizon)

        try:
            df_p = ts[[date_col, kpi_col]].rename(columns={date_col: "ds", kpi_col: "y"})
            df_p["ds"] = pd.to_datetime(df_p["ds"])
            df_p = df_p.dropna()

            m = Prophet(
                yearly_seasonality="auto",
                weekly_seasonality="auto",
                daily_seasonality=False,
                interval_width=0.80,
                changepoint_prior_scale=0.05,
            )
            m.fit(df_p)

            future = m.make_future_dataframe(periods=horizon)
            forecast = m.predict(future)

            forecast_only = forecast[forecast["ds"] > df_p["ds"].max()][
                ["ds", "yhat", "yhat_lower", "yhat_upper"]
            ].reset_index(drop=True)

            last_actual = df_p["y"].iloc[-1]
            last_forecast = forecast_only["yhat"].iloc[-1]
            direction = "up" if last_forecast > last_actual else "down"
            pct = abs((last_forecast - last_actual) / last_actual * 100) if last_actual else 0

            return AgentResult(
                agent=self.name,
                status="success",
                summary=(
                    f"Prophet forecast ({horizon} periods): {kpi_col} expected to go "
                    f"{direction} by {pct:.1f}% "
                    f"(from {last_actual:,.1f} to {last_forecast:,.1f}). "
                    f"80% CI: [{forecast_only['yhat_lower'].iloc[-1]:,.1f}, "
                    f"{forecast_only['yhat_upper'].iloc[-1]:,.1f}]."
                ),
                data={
                    "forecast_df": forecast_only,
                    "full_forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                    "horizon": horizon,
                    "method": "Prophet",
                    "direction": direction,
                    "pct_change": round(pct, 2),
                    "last_actual": round(last_actual, 2),
                    "last_forecast": round(last_forecast, 2),
                },
            )
        except Exception as e:
            self.logger.warning(f"Prophet failed: {e} — falling back to ARIMA")
            return self._arima(ts, date_col, kpi_col, horizon)

    # ------------------------------------------------------------------
    # ARIMA
    # ------------------------------------------------------------------
    def _arima(self, ts, date_col, kpi_col, horizon) -> AgentResult:
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            return self._ets(ts, date_col, kpi_col, horizon)

        try:
            series = ts[kpi_col].dropna().values
            model = ARIMA(series, order=(1, 1, 1))
            fit = model.fit()
            forecast_result = fit.get_forecast(steps=horizon)
            mean = forecast_result.predicted_mean
            ci = forecast_result.conf_int(alpha=0.20)

            last_date = pd.to_datetime(ts[date_col].iloc[-1])
            freq = pd.infer_freq(pd.to_datetime(ts[date_col])) or "D"
            future_dates = pd.date_range(last_date, periods=horizon + 1, freq=freq)[1:]

            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "yhat": mean,
                "yhat_lower": ci.iloc[:, 0].values,
                "yhat_upper": ci.iloc[:, 1].values,
            })

            last_actual = float(series[-1])
            last_forecast = float(mean[-1])
            direction = "up" if last_forecast > last_actual else "down"
            pct = abs((last_forecast - last_actual) / last_actual * 100) if last_actual else 0

            return AgentResult(
                agent=self.name,
                status="success",
                summary=(
                    f"ARIMA(1,1,1) forecast ({horizon} periods): {kpi_col} trending "
                    f"{direction} by ~{pct:.1f}% "
                    f"(from {last_actual:,.1f} to {last_forecast:,.1f})."
                ),
                data={
                    "forecast_df": forecast_df,
                    "horizon": horizon,
                    "method": "ARIMA",
                    "direction": direction,
                    "pct_change": round(pct, 2),
                    "last_actual": round(last_actual, 2),
                    "last_forecast": round(last_forecast, 2),
                },
            )
        except Exception as e:
            self.logger.warning(f"ARIMA failed: {e} — falling back to ETS")
            return self._ets(ts, date_col, kpi_col, horizon)

    # ------------------------------------------------------------------
    # ETS / Holt's linear (lightweight fallback)
    # ------------------------------------------------------------------
    def _ets(self, ts, date_col, kpi_col, horizon) -> AgentResult:
        try:
            series = ts[kpi_col].dropna().values.astype(float)
            if len(series) < 3:
                return self.skip("Insufficient data for any forecasting method.")

            # Holt's linear trend
            alpha, beta = 0.3, 0.1
            level = [series[0]]
            trend = [series[1] - series[0]]
            for val in series[1:]:
                prev_l, prev_t = level[-1], trend[-1]
                l = alpha * val + (1 - alpha) * (prev_l + prev_t)
                t = beta * (l - prev_l) + (1 - beta) * prev_t
                level.append(l)
                trend.append(t)

            forecasts = [level[-1] + i * trend[-1] for i in range(1, horizon + 1)]

            last_date = pd.to_datetime(ts[date_col].iloc[-1])
            freq = "D"
            future_dates = pd.date_range(last_date, periods=horizon + 1, freq=freq)[1:]

            forecast_df = pd.DataFrame({
                "ds": future_dates,
                "yhat": forecasts,
                "yhat_lower": [f * 0.9 for f in forecasts],
                "yhat_upper": [f * 1.1 for f in forecasts],
            })

            last_actual = float(series[-1])
            last_forecast = float(forecasts[-1])
            direction = "up" if last_forecast > last_actual else "down"
            pct = abs((last_forecast - last_actual) / last_actual * 100) if last_actual else 0

            return AgentResult(
                agent=self.name,
                status="success",
                summary=(
                    f"Holt linear forecast ({horizon} periods): {kpi_col} expected "
                    f"{direction} by ~{pct:.1f}% "
                    f"(from {last_actual:,.1f} to {last_forecast:,.1f})."
                ),
                data={
                    "forecast_df": forecast_df,
                    "horizon": horizon,
                    "method": "Holt ETS",
                    "direction": direction,
                    "pct_change": round(pct, 2),
                    "last_actual": round(last_actual, 2),
                    "last_forecast": round(last_forecast, 2),
                },
            )
        except Exception as e:
            return AgentResult(
                agent=self.name, status="error",
                summary=f"All forecast methods failed: {e}",
                data={}, error=str(e),
            )
