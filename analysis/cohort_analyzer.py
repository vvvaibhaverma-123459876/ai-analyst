"""
analysis/cohort_analyzer.py
Cohort analysis: signup cohort retention, time-to-convert, D1/D7/D30.
"""

import pandas as pd
import numpy as np
from core.logger import get_logger

logger = get_logger(__name__)


class CohortAnalyzer:

    def build_retention_matrix(
        self,
        df: pd.DataFrame,
        user_col: str,
        date_col: str,
        cohort_grain: str = "M",
    ) -> pd.DataFrame:
        """
        Standard cohort retention matrix.
        Each cell = % of cohort still active at period N.

        Args:
            df: event/activity DataFrame (one row per user-date activity)
            user_col: user identifier column
            date_col: activity date column
            cohort_grain: 'D' | 'W' | 'M'

        Returns:
            pivot table: cohort_period (index) × period_number (columns) → retention %
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        df["cohort_period"] = df.groupby(user_col)[date_col].transform("min").dt.to_period(cohort_grain)
        df["activity_period"] = df[date_col].dt.to_period(cohort_grain)
        df["period_number"] = (df["activity_period"] - df["cohort_period"]).apply(lambda x: x.n)

        cohort_sizes = df.groupby("cohort_period")[user_col].nunique().rename("cohort_size")

        retention = df.groupby(["cohort_period", "period_number"])[user_col].nunique().reset_index()
        retention = retention.merge(cohort_sizes, on="cohort_period")
        retention["retention_pct"] = (retention[user_col] / retention["cohort_size"] * 100).round(1)

        matrix = retention.pivot_table(
            index="cohort_period",
            columns="period_number",
            values="retention_pct",
        )
        matrix.columns = [f"Period {c}" if c > 0 else "Cohort Size" for c in matrix.columns]

        logger.info(f"Retention matrix: {matrix.shape[0]} cohorts × {matrix.shape[1]} periods")
        return matrix

    def time_to_convert(
        self,
        df: pd.DataFrame,
        user_col: str,
        signup_event: str,
        convert_event: str,
        event_col: str,
        date_col: str,
    ) -> pd.DataFrame:
        """
        Computes days between signup and conversion for each user.

        Returns:
            DataFrame with [user_col, signup_date, convert_date, days_to_convert]
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])

        signups = df[df[event_col] == signup_event].groupby(user_col)[date_col].min().rename("signup_date")
        converts = df[df[event_col] == convert_event].groupby(user_col)[date_col].min().rename("convert_date")

        merged = pd.concat([signups, converts], axis=1).dropna()
        merged["days_to_convert"] = (merged["convert_date"] - merged["signup_date"]).dt.days
        merged = merged[merged["days_to_convert"] >= 0].reset_index()

        logger.info(f"Time-to-convert: {len(merged)} converted users, "
                    f"median={merged['days_to_convert'].median():.1f} days")
        return merged

    def cohort_kpi_summary(
        self,
        df: pd.DataFrame,
        user_col: str,
        date_col: str,
        kpi_col: str,
        cohort_grain: str = "M",
        periods: list[int] = None,
    ) -> pd.DataFrame:
        """
        Average KPI value per cohort at period D1, D7, D30 (or custom periods).
        """
        periods = periods or [1, 7, 30]
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["cohort_date"] = df.groupby(user_col)[date_col].transform("min").dt.to_period(cohort_grain)
        df["activity_date"] = df[date_col].dt.to_period(cohort_grain)
        df["period_number"] = (df["activity_date"] - df["cohort_date"]).apply(lambda x: x.n)

        result = (
            df[df["period_number"].isin(periods)]
            .groupby(["cohort_date", "period_number"])[kpi_col]
            .mean()
            .round(2)
            .reset_index()
        )
        pivot = result.pivot(index="cohort_date", columns="period_number", values=kpi_col)
        pivot.columns = [f"D{c}" for c in pivot.columns]
        return pivot
