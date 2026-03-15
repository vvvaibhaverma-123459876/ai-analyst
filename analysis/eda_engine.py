"""
analysis/eda_engine.py
Auto exploratory data analysis.
profile_dataframe() is preserved from app.py v0.1 and extended.
"""

import pandas as pd
import numpy as np
from core.logger import get_logger

logger = get_logger(__name__)


class EDAEngine:

    def profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extended version of profile_dataframe() from app.py.
        Returns per-column stats table.
        """
        rows = []
        for c in df.columns:
            col = df[c]
            is_numeric = pd.api.types.is_numeric_dtype(col)
            row = {
                "column": c,
                "dtype": str(col.dtype),
                "nulls": int(col.isna().sum()),
                "null_pct": round(col.isna().mean() * 100, 1),
                "unique": int(col.nunique()),
            }
            if is_numeric:
                row.update({
                    "mean": round(col.mean(), 2),
                    "std": round(col.std(), 2),
                    "min": round(col.min(), 2),
                    "p25": round(col.quantile(0.25), 2),
                    "median": round(col.median(), 2),
                    "p75": round(col.quantile(0.75), 2),
                    "max": round(col.max(), 2),
                    "skew": round(col.skew(), 2),
                })
            else:
                top = col.value_counts().index[0] if col.notna().any() else None
                row.update({
                    "top_value": str(top),
                    "top_freq": int(col.value_counts().iloc[0]) if col.notna().any() else 0,
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def infer_kpis(self, df: pd.DataFrame) -> list[str]:
        """Return numeric columns that are likely KPIs (not IDs)."""
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [
            c for c in numeric
            if not any(kw in c.lower() for kw in ["id", "index", "rank", "seq"])
        ]

    def infer_dimensions(self, df: pd.DataFrame) -> list[str]:
        """Return categorical columns with reasonable cardinality."""
        return [
            c for c in df.select_dtypes(include=["object", "category"]).columns
            if df[c].nunique() <= 50
        ]

    def quality_report(self, df: pd.DataFrame) -> dict:
        """High-level quality summary."""
        total = len(df)
        return {
            "total_rows": total,
            "total_columns": len(df.columns),
            "complete_rows": int(df.dropna().shape[0]),
            "completeness_pct": round(df.dropna().shape[0] / total * 100, 1) if total else 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=["object", "category"]).columns),
            "datetime_columns": len(df.select_dtypes(include=["datetime"]).columns),
        }
