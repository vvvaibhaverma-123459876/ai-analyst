import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from openai import OpenAI

st.set_page_config(page_title="AI Analyst v0.1", layout="wide")
st.title("AI Analyst v0.1")
st.caption("Upload a CSV → Profile → Trend → Anomalies → Drivers → Executive Summary")

# ------------------------------
# Helper Functions
# ------------------------------

def detect_datetime_column(df: pd.DataFrame):
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            return c

    best_col = None
    best_valid = 0
    for c in df.columns:
        if df[c].dtype == "object":
            parsed = pd.to_datetime(df[c], errors="coerce")
            valid = parsed.notna().sum()
            if valid > best_valid and valid >= int(0.6 * len(df)):
                best_valid = valid
                best_col = c
    return best_col

def profile_dataframe(df: pd.DataFrame):
    return pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "nulls": [int(df[c].isna().sum()) for c in df.columns],
        "unique": [int(df[c].nunique()) for c in df.columns],
    })

def driver_attribution(df, date_col, kpi_col, days=3):
    dfx = df.copy()
    dfx["__date__"] = dfx[date_col].dt.date
    dfx["__date__"] = pd.to_datetime(dfx["__date__"])

    max_date = dfx["__date__"].max()
    start_last = max_date - pd.Timedelta(days=days - 1)
    start_prev = start_last - pd.Timedelta(days=days)

    last = dfx[(dfx["__date__"] >= start_last)]
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
        merged["dimension"] = c
        merged["value"] = merged.index
        merged = merged.reset_index(drop=True)[["dimension", "value", "prev", "last", "delta"]]
        results.append(merged)

    drivers = pd.concat(results).sort_values("delta") if results else pd.DataFrame()

    return delta, pct, drivers

def generate_exec_summary(payload):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set."

    client = OpenAI(api_key=api_key)

    system = "You are a senior analytics lead. Use only provided facts."
    user = f"""
    Write an executive summary with sections:
    1) What happened
    2) Why (evidence-based)
    3) Impact
    4) Recommended actions

    Facts:
    {payload}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

# ------------------------------
# App Starts Here
# ------------------------------

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is None:
    st.stop()

df = pd.read_csv(uploaded)

st.subheader("Preview")
st.dataframe(df.head())

st.subheader("Data Profile")
st.dataframe(profile_dataframe(df))

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
date_guess = detect_datetime_column(df)

col1, col2 = st.columns(2)
with col1:
    date_col = st.selectbox("Select date column", df.columns, index=df.columns.tolist().index(date_guess) if date_guess else 0)
with col2:
    kpi_col = st.selectbox("Select KPI", numeric_cols)

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df_valid = df[df[date_col].notna()].copy()

grain = st.radio("Time Grain", ["Daily", "Weekly", "Monthly"], horizontal=True)

df_valid["__date__"] = df_valid[date_col].dt.date
ts = df_valid.groupby("__date__")[kpi_col].sum().reset_index()
ts["__date__"] = pd.to_datetime(ts["__date__"])

if grain == "Weekly":
    ts = ts.set_index("__date__").resample("W")[kpi_col].sum().reset_index()
elif grain == "Monthly":
    ts = ts.set_index("__date__").resample("M")[kpi_col].sum().reset_index()

# ------------------------------
# Anomaly Detection
# ------------------------------

window = st.slider("Rolling window", 3, 30, 7)
z_thresh = st.slider("Z-score threshold", 1.0, 5.0, 2.0)

ts["mean"] = ts[kpi_col].rolling(window).mean()
ts["std"] = ts[kpi_col].rolling(window).std()
ts["zscore"] = (ts[kpi_col] - ts["mean"]) / ts["std"]
ts["anomaly"] = ts["zscore"].abs() > z_thresh

fig, ax = plt.subplots()
ax.plot(ts["__date__"], ts[kpi_col])
ax.scatter(ts[ts["anomaly"]]["__date__"],
           ts[ts["anomaly"]][kpi_col],
           color="red")
plt.xticks(rotation=45)
st.pyplot(fig)

anoms = ts[ts["anomaly"]]

# ------------------------------
# Driver Attribution
# ------------------------------

st.subheader("Driver Attribution")

days = st.slider("Compare last N days", 2, 14, 3)
delta, pct, drivers = driver_attribution(df_valid, date_col, kpi_col, days)

st.metric("Overall Change", f"{delta:,.2f}", f"{pct:.2f}%")

if not drivers.empty:
    st.write("Top Negative Drivers")
    st.dataframe(drivers.head(3))
    st.write("Top Positive Drivers")
    st.dataframe(drivers.tail(3).sort_values("delta", ascending=False))

# ------------------------------
# Executive Summary
# ------------------------------

st.subheader("Executive Summary")

payload = {
    "kpi": kpi_col,
    "delta": delta,
    "pct_change": pct,
    "anomalies": anoms[["__date__", kpi_col]].to_dict("records"),
    "top_drivers": drivers.head(3).to_dict("records")
}

if st.button("Generate Executive Summary"):
    summary = generate_exec_summary(payload)
    st.markdown(summary)