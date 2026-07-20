from __future__ import annotations
from dataclasses import dataclass, field, asdict
import pandas as pd

_UNIX_EPOCH = pd.Timestamp("1970-01-01")


def _is_degenerate_epoch_parse(parsed: pd.Series, row_count: int) -> bool:
    """True if parsed datetimes are the unmistakable signature of raw
    numbers being reinterpreted as nanoseconds/seconds/etc since the Unix
    epoch by pd.to_datetime, rather than real per-row dates — not "sparse"
    or "low quality" dates, but not dates at all. Two independent tells:
    every parsed value landing within a minute of 1970-01-01, or a single
    unique timestamp across a multi-thousand-row frame (a constant/near-
    constant numeric column, e.g. a count or flag, parsed as "a date")."""
    if (parsed.max() - _UNIX_EPOCH) < pd.Timedelta(seconds=60) and (parsed.min() - _UNIX_EPOCH) > pd.Timedelta(seconds=-60):
        return True
    if row_count >= 1000 and parsed.nunique() <= 1:
        return True
    return False


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
    # Distinct from blocking_reasons: a fatal setup error (the date column
    # isn't a date at all) rather than a quality gradient. GovernedPipeline
    # raises DataQualityError and halts when this is non-empty, instead of
    # the usual "log a warning, continue with downgraded confidence".
    fatal_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

class DataQualityGate:
    def assess(self, df: pd.DataFrame, date_col: str = '', kpi_col: str = '') -> DataQualityReport:
        warnings: list[str] = []
        blocking: list[str] = []
        fatal: list[str] = []
        row_count = len(df)
        if row_count == 0:
            return DataQualityReport(False, 0.0, False, False, False, False, 0.0, 1.0, 0, ['empty dataframe'], ['empty dataframe'], [])
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
        if kpi_col and kpi_col in df.columns:
            kpi_null_ratio = float(df[kpi_col].isna().mean())
            if kpi_null_ratio > 0.20:
                warnings.append(f'elevated KPI null ratio: {kpi_null_ratio:.1%}')
            if kpi_null_ratio >= 0.80:
                blocking.append('KPI null ratio too high')
            completeness_ok = completeness_ok and kpi_null_ratio < 0.80
        freshness_ok = True
        continuity_ok = True
        if date_col and date_col in df.columns:
            raw_col = df[date_col]
            is_numeric_source = pd.api.types.is_numeric_dtype(raw_col)
            dt = pd.to_datetime(raw_col, errors='coerce').dropna().sort_values()
            if dt.empty:
                freshness_ok = False
                continuity_ok = False
                blocking.append('date parsing failed')
            elif is_numeric_source and _is_degenerate_epoch_parse(dt, row_count):
                freshness_ok = False
                continuity_ok = False
                message = (
                    f"'{date_col}' is not a date column — it is numeric, and parsing it as "
                    f"a date collapses to {dt.nunique()} unique timestamp(s) between "
                    f"{dt.min()} and {dt.max()} across {row_count} rows. Choose an actual "
                    "date/time column."
                )
                blocking.append(message)
                fatal.append(message)
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
        if kpi_col and kpi_col in df.columns:
            score -= min(float(df[kpi_col].isna().mean()), 0.9) * 0.55
        if blocking:
            score = min(score, 0.35)
        score = max(0.0, round(score, 3))
        ok = not blocking
        return DataQualityReport(ok, score, freshness_ok, completeness_ok, continuity_ok, sufficiency_ok, round(duplicate_ratio, 4), round(null_ratio, 4), row_count, warnings, blocking, fatal)
