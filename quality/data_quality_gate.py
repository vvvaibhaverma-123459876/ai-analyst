from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd

@dataclass
class DataQualityReport:
    ok: bool
    score: float
    freshness_ok: bool
    completeness_ok: bool
    continuity_ok: bool
    sufficiency_ok: bool
    duplicate_ratio: float
    null_ratio: float
    row_count: int
    warnings: list[str]
    blocking_reasons: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

class DataQualityGate:
    def assess(self, df: pd.DataFrame, date_col: str = '', kpi_col: str = '') -> DataQualityReport:
        warnings: list[str] = []
        blocking: list[str] = []
        row_count = len(df)
        if row_count == 0:
            return DataQualityReport(False, 0.0, False, False, False, False, 0.0, 1.0, 0, ['empty dataframe'], ['empty dataframe'])
        duplicate_ratio = float(df.duplicated().mean()) if row_count else 0.0
        if duplicate_ratio > 0.25:
            warnings.append(f'high duplicate ratio: {duplicate_ratio:.1%}')
        if duplicate_ratio > 0.60:
            blocking.append('duplicate ratio too high')
        null_ratio = float(df.isna().mean().mean()) if row_count else 1.0
        if null_ratio > 0.20:
            warnings.append(f'elevated null ratio: {null_ratio:.1%}')
        if null_ratio > 0.70:
            blocking.append('null ratio too high')
        completeness_ok = kpi_col in df.columns if kpi_col else True
        if kpi_col and not completeness_ok:
            blocking.append(f'missing KPI column: {kpi_col}')
        freshness_ok = True
        continuity_ok = True
        if date_col and date_col in df.columns:
            dt = pd.to_datetime(df[date_col], errors='coerce').dropna().sort_values()
            if dt.empty:
                freshness_ok = False
                continuity_ok = False
                blocking.append('date parsing failed')
            else:
                span_days = max((dt.max() - dt.min()).days, 0)
                if len(dt) >= 3 and span_days > 0:
                    continuity_ratio = dt.nunique() / (span_days + 1)
                    continuity_ok = continuity_ratio >= 0.25
                    if not continuity_ok:
                        warnings.append('time continuity appears sparse')
        elif date_col:
            warnings.append(f'date column not found: {date_col}')
            freshness_ok = False
            continuity_ok = False
        sufficiency_ok = row_count >= 10
        if row_count < 10:
            warnings.append('very small sample size')
        if row_count < 3:
            blocking.append('insufficient rows for analysis')
        score = 1.0
        score -= min(duplicate_ratio, 0.5) * 0.35
        score -= min(null_ratio, 0.8) * 0.35
        score -= 0.15 if not continuity_ok else 0.0
        score -= 0.15 if not sufficiency_ok else 0.0
        if blocking:
            score = min(score, 0.35)
        score = max(0.0, round(score, 3))
        ok = not blocking
        return DataQualityReport(ok, score, freshness_ok, completeness_ok, continuity_ok, sufficiency_ok, round(duplicate_ratio, 4), round(null_ratio, 4), row_count, warnings, blocking)
