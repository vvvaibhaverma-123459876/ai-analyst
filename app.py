import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Analyst v0.1", layout="wide")
st.title("AI Analyst v0.1")
st.caption("Upload a CSV → profile → pick KPI → view trend")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

def detect_datetime_column(df: pd.DataFrame):
    """Return best-guess datetime column name or None."""
    # 1) If any column already datetime
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c

    # 2) Try parsing object columns and pick the one with most valid dates
    best_col = None
    best_valid = 0
    for c in df.columns:
        if df[c].dtype == "object":
            parsed = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
            valid = parsed.notna().sum()
            # Require at least 60% parse success to be considered
            if valid > best_valid and valid >= int(0.6 * len(df)):
                best_valid = valid
                best_col = c
    return best_col

def driver_attribution(df: pd.DataFrame, date_col: str, kpi_col: str, days: int = 3):
    """
    Compare last N days vs previous N days.
    Returns a dict with overall delta and driver tables for each categorical column.
    """
    dfx = df.copy()
    dfx = dfx[dfx[date_col].notna()].copy()
    dfx["__date__"] = dfx[date_col].dt.date
    dfx["__date__"] = pd.to_datetime(dfx["__date__"])

    max_date = dfx["__date__"].max()
    start_last = max_date - pd.Timedelta(days=days - 1)
    start_prev = start_last - pd.Timedelta(days=days)

    last = dfx[(dfx["__date__"] >= start_last) & (dfx["__date__"] <= max_date)]
    prev = dfx[(dfx["__date__"] >= start_prev) & (dfx["__date__"] < start_last)]

    last_total = last[kpi_col].sum()
    prev_total = prev[kpi_col].sum()
    overall_delta = last_total - prev_total
    pct = (overall_delta / prev_total * 100) if prev_total != 0 else np.nan

    # Categorical columns = object dtype (excluding date col)
    cat_cols = [c for c in dfx.columns if dfx[c].dtype == "object" and c != date_col]

    results = []
    for c in cat_cols:
        last_g = last.groupby(c)[kpi_col].sum().rename("last")
        prev_g = prev.groupby(c)[kpi_col].sum().rename("prev")
        merged = pd.concat([last_g, prev_g], axis=1).fillna(0)
        merged["delta"] = merged["last"] - merged["prev"]
        merged["dimension"] = c
        merged["value"] = merged.index
        merged = merged.reset_index(drop=True)[["dimension", "value", "prev", "last", "delta"]]
        results.append(merged)

    drivers = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if not drivers.empty:
        drivers = drivers.sort_values("delta")

    return {
        "last_total": last_total,
        "prev_total": prev_total,
        "overall_delta": overall_delta,
        "overall_pct": pct,
        "drivers": drivers,
        "periods": {
            "prev_start": start_prev.date(),
            "prev_end": (start_last - pd.Timedelta(days=1)).date(),
            "last_start": start_last.date(),
            "last_end": max_date.date(),
        }
    }

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prof = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "nulls": [int(df[c].isna().sum()) for c in df.columns],
        "null_%": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
        "unique": [int(df[c].nunique(dropna=True)) for c in df.columns],
        "example": [df[c].dropna().iloc[0] if df[c].dropna().shape[0] else None for c in df.columns],
    })
    return prof

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Read CSV
df = pd.read_csv(uploaded)

st.subheader("Preview")
st.dataframe(df.head(50), use_container_width=True)

st.subheader("Shape")
st.write({"rows": df.shape[0], "columns": df.shape[1]})

# ---- Profiling ----
st.subheader("Data Profile")
prof = profile_dataframe(df)
st.dataframe(prof, use_container_width=True)

# Numeric summary
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if numeric_cols:
    st.subheader("Numeric Summary")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
else:
    st.warning("No numeric columns found. Add numeric KPIs like signups/revenue.")
    st.stop()

# ---- Date + KPI detection ----
st.subheader("KPI Trend")

date_col_guess = detect_datetime_column(df)

left, right = st.columns(2)
with left:
    date_col = st.selectbox(
        "Select date/time column",
        options=[None] + df.columns.tolist(),
        index=(df.columns.tolist().index(date_col_guess) + 1) if date_col_guess in df.columns else 0
    )

with right:
    kpi_col = st.selectbox("Select KPI (numeric column)", options=numeric_cols)

if date_col is None:
    st.warning("Select a date/time column to plot a trend.")
    st.stop()

# Parse datetime
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
df_valid = df[df[date_col].notna()].copy()

if df_valid.empty:
    st.error("Could not parse any valid dates in the selected date column.")
    st.stop()

# Choose time grain
grain = st.radio("Time grain", options=["Daily", "Weekly", "Monthly"], horizontal=True)

# Aggregate
df_valid = df_valid.sort_values(date_col)
df_valid["__date__"] = df_valid[date_col].dt.date
ts = df_valid.groupby("__date__")[kpi_col].sum().reset_index()
ts["__date__"] = pd.to_datetime(ts["__date__"])

if grain == "Weekly":
    ts = ts.set_index("__date__").resample("W")[kpi_col].sum().reset_index()
elif grain == "Monthly":
    ts = ts.set_index("__date__").resample("M")[kpi_col].sum().reset_index()

# ---- Anomaly Detection ----
st.subheader("Anomaly Detection")

st.subheader("Driver Attribution (What changed?)")

days = st.slider("Compare last N days vs previous N days", min_value=2, max_value=14, value=3)

attrib = driver_attribution(df_valid, date_col, kpi_col, days=days)

st.write(
    f"Comparing **{attrib['periods']['last_start']} → {attrib['periods']['last_end']}** "
    f"vs **{attrib['periods']['prev_start']} → {attrib['periods']['prev_end']}**"
)

st.metric(
    "Total change",
    value=f"{attrib['last_total']:,.2f}",
    delta=f"{attrib['overall_delta']:,.2f} ({attrib['overall_pct']:.2f}%)"
)

drivers = attrib["drivers"]
if drivers is None or drivers.empty:
    st.info("No categorical columns found for driver attribution.")
else:
    st.write("Top negative drivers (contributed to decrease):")
    st.dataframe(drivers.head(3), use_container_width=True)

    st.write("Top positive drivers (contributed to increase):")
    st.dataframe(drivers.tail(3).sort_values("delta", ascending=False), use_container_width=True)

window = st.slider("Rolling window (points)", min_value=3, max_value=30, value=min(7, max(3, len(ts)//2)))
z_thresh = st.slider("Z-score threshold", min_value=1.0, max_value=5.0, value=2.0, step=0.1)

ts2 = ts.copy()
ts2["rolling_mean"] = ts2[kpi_col].rolling(window=window, min_periods=3).mean()
ts2["rolling_std"] = ts2[kpi_col].rolling(window=window, min_periods=3).std()

# Avoid divide-by-zero
ts2["zscore"] = (ts2[kpi_col] - ts2["rolling_mean"]) / ts2["rolling_std"].replace(0, np.nan)
ts2["is_anomaly"] = ts2["zscore"].abs() >= z_thresh

# Plot with anomalies
fig, ax = plt.subplots()
ax.plot(ts2["__date__"], ts2[kpi_col], label="KPI")

anoms = ts2[ts2["is_anomaly"] & ts2["zscore"].notna()]
if not anoms.empty:
    ax.scatter(anoms["__date__"], anoms[kpi_col], color="red", label="Anomaly")

ax.set_title(f"{kpi_col} trend ({grain.lower()}) with anomalies")
ax.set_xlabel("Date")
ax.set_ylabel(kpi_col)
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Anomaly table
if not anoms.empty:
    st.write("Detected anomalies:")
    out = anoms[["__date__", kpi_col, "zscore"]].copy()
    out["zscore"] = out["zscore"].round(2)
    st.dataframe(out, use_container_width=True)
else:
    st.info("No anomalies detected with current settings.")

# ---- Quick delta ----
st.subheader("Latest change")
if len(ts2) >= 2:
    latest = ts2[kpi_col].iloc[-1]
    prev = ts2[kpi_col].iloc[-2]
    delta = latest - prev
    pct = (delta / prev * 100) if prev != 0 else np.nan
    st.metric(label="Latest vs Previous", value=f"{latest:,.2f}", delta=f"{delta:,.2f} ({pct:,.2f}%)")