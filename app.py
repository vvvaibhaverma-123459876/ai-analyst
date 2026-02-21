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

# Plot
fig, ax = plt.subplots()
ax.plot(ts["__date__"], ts[kpi_col])
ax.set_title(f"{kpi_col} trend ({grain.lower()})")
ax.set_xlabel("Date")
ax.set_ylabel(kpi_col)
plt.xticks(rotation=45)
st.pyplot(fig)

# Quick delta
if len(ts) >= 2:
    latest = ts[kpi_col].iloc[-1]
    prev = ts[kpi_col].iloc[-2]
    delta = latest - prev
    pct = (delta / prev * 100) if prev != 0 else np.nan
    st.metric(label="Latest vs Previous", value=f"{latest:,.2f}", delta=f"{delta:,.2f} ({pct:,.2f}%)")