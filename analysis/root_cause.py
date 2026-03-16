"""
analysis/root_cause.py  — v10
Root cause analysis: driver attribution + contribution analysis.

v10: implements AnalysisContract — analyze() is the canonical entry point.
     All existing methods preserved for backward-compat with agents.
"""

import pandas as pd
import numpy as np
from analysis.contract import AnalysisContract, AnalysisResult
from core.config import config
from core.logger import get_logger

logger = get_logger(__name__)


class RootCauseAnalyzer(AnalysisContract):

    module_name = "root_cause"

    # ------------------------------------------------------------------
    # AnalysisContract: canonical entry point
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, **kwargs) -> AnalysisResult:
        """
        Contract entry point.  kwargs accepted:
          date_col, kpi_col (or value_col alias), days
        Returns AnalysisResult with drivers DataFrame in metadata["drivers"].
        """
        date_col = kwargs.get("date_col", "")
        kpi_col  = kwargs.get("kpi_col") or kwargs.get("value_col", "")
        days     = kwargs.get("days")

        required = [c for c in [date_col, kpi_col] if c]
        errors = self.validate_inputs(df, required_cols=required, min_rows=6)
        if errors:
            return AnalysisResult(
                module=self.module_name, ok=False, errors=errors,
                summary="Root cause analysis could not run.", method="driver_attribution",
            )

        try:
            result = self.driver_attribution(df, date_col, kpi_col, days)
            drivers  = result["drivers"]
            movers   = self.top_movers(drivers, n=5)
            delta    = result["delta"]
            pct      = result["pct_change"]
            direction = "up" if delta >= 0 else "down"
            neg_drivers = movers.get("negative", [])
            top_neg = neg_drivers[0] if neg_drivers else {}

            summary = (
                f"{kpi_col} {direction} {abs(pct):.1f}% "
                f"(delta={delta:+.0f})."
            )
            if top_neg:
                summary += (
                    f" Primary driver: {top_neg.get('dimension')}="
                    f"{top_neg.get('value')} "
                    f"({top_neg.get('pct_contribution',0):+.1f}% contribution)."
                )

            return AnalysisResult(
                module=self.module_name,
                ok=True,
                records=drivers.to_dict("records") if not drivers.empty else [],
                summary=summary,
                confidence=1.0,
                method="driver_attribution",
                metadata={
                    "drivers":    drivers,
                    "movers":     movers,
                    "delta":      delta,
                    "pct_change": pct,
                    "last_total": result["last_total"],
                    "prev_total": result["prev_total"],
                    "period_last": result["period_last"],
                    "period_prev": result["period_prev"],
                },
            )
        except Exception as e:
            return AnalysisResult(
                module=self.module_name, ok=False,
                errors=[str(e)], summary=f"Root cause error: {e}",
                method="driver_attribution",
            )

    # ------------------------------------------------------------------
    # Driver Attribution (preserved + extended from app.py)
    # ------------------------------------------------------------------

    def driver_attribution(
        self,
        df: pd.DataFrame,
        date_col: str,
        kpi_col: str,
        days: int = None,
    ) -> dict:
        """
        Compare last N days vs previous N days across all categorical dimensions.
        Extracted and extended from driver_attribution() in app.py v0.1.

        Returns:
            {
                delta: float,
                pct_change: float,
                last_total: float,
                prev_total: float,
                drivers: pd.DataFrame,
                period_last: (start, end),
                period_prev: (start, end),
            }
        """
        n = days or config.DEFAULT_DRIVER_DAYS
        dfx = df.copy()
        dfx["__date__"] = pd.to_datetime(dfx[date_col].dt.date)

        max_date = dfx["__date__"].max()
        start_last = max_date - pd.Timedelta(days=n - 1)
        start_prev = start_last - pd.Timedelta(days=n)

        last = dfx[dfx["__date__"] >= start_last]
        prev = dfx[(dfx["__date__"] < start_last) & (dfx["__date__"] >= start_prev)]

        last_total = last[kpi_col].sum()
        prev_total = prev[kpi_col].sum()
        delta = last_total - prev_total
        pct = (delta / prev_total * 100) if prev_total != 0 else 0

        cat_cols = [c for c in dfx.columns if dfx[c].dtype == "object"]
        results = []
        for c in cat_cols:
            last_g = last.groupby(c)[kpi_col].sum()
            prev_g = prev.groupby(c)[kpi_col].sum()
            merged = pd.concat([last_g, prev_g], axis=1).fillna(0)
            merged.columns = ["last", "prev"]
            merged["delta"] = merged["last"] - merged["prev"]
            merged["pct_contribution"] = (
                (merged["delta"] / abs(delta) * 100).round(1) if delta != 0 else 0
            )
            merged["dimension"] = c
            merged["value"] = merged.index
            merged = merged.reset_index(drop=True)[
                ["dimension", "value", "prev", "last", "delta", "pct_contribution"]
            ]
            results.append(merged)

        drivers = pd.concat(results).sort_values("delta") if results else pd.DataFrame()

        logger.info(f"Driver attribution: delta={delta:+.2f} ({pct:+.1f}%), {len(drivers)} driver rows")

        return {
            "delta": delta,
            "pct_change": pct,
            "last_total": last_total,
            "prev_total": prev_total,
            "drivers": drivers,
            "period_last": (start_last.date(), max_date.date()),
            "period_prev": (start_prev.date(), (start_last - pd.Timedelta(days=1)).date()),
        }

    # ------------------------------------------------------------------
    # Contribution Analysis (new)
    # ------------------------------------------------------------------

    def contribution_analysis(
        self,
        df: pd.DataFrame,
        kpi_col: str,
        dim_col: str,
        date_col: str = None,
        period_col: str = None,
    ) -> pd.DataFrame:
        """
        Computes each segment's share of total KPI.
        If date_col provided, computes share shift over time.

        Returns DataFrame with columns:
            [dim_col, kpi_total, share_pct, share_change_pp]
        """
        grouped = df.groupby(dim_col)[kpi_col].sum().reset_index()
        total = grouped[kpi_col].sum()
        grouped["share_pct"] = (grouped[kpi_col] / total * 100).round(2)
        grouped = grouped.sort_values("share_pct", ascending=False)

        if date_col:
            # Compare first half vs second half of the data
            df = df.copy()
            df["__date__"] = pd.to_datetime(df[date_col])
            mid = df["__date__"].median()

            first = df[df["__date__"] <= mid].groupby(dim_col)[kpi_col].sum()
            second = df[df["__date__"] > mid].groupby(dim_col)[kpi_col].sum()

            first_share = (first / first.sum() * 100).rename("share_first")
            second_share = (second / second.sum() * 100).rename("share_second")

            shift = pd.concat([first_share, second_share], axis=1).fillna(0)
            shift["share_change_pp"] = (shift["share_second"] - shift["share_first"]).round(2)
            grouped = grouped.merge(
                shift[["share_change_pp"]].reset_index(),
                on=dim_col, how="left"
            )

        logger.info(f"Contribution analysis: {len(grouped)} segments for '{dim_col}'")
        return grouped

    # ------------------------------------------------------------------
    # Top movers summary
    # ------------------------------------------------------------------

    def top_movers(self, drivers: pd.DataFrame, n: int = 3) -> dict:
        """Returns top positive and negative driver rows as dicts."""
        if drivers.empty:
            return {"positive": [], "negative": []}
        pos = drivers.nlargest(n, "delta")[["dimension", "value", "delta", "pct_contribution"]]
        neg = drivers.nsmallest(n, "delta")[["dimension", "value", "delta", "pct_contribution"]]
        return {
            "positive": pos.to_dict("records"),
            "negative": neg.to_dict("records"),
        }
