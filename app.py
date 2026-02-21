import streamlit as st
import pandas as pd

st.set_page_config(page_title="AI Analyst v0.1", layout="wide")
st.title("AI Analyst v0.1")
st.caption("Upload a CSV → preview data → (more insights coming in Block 2)")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Shape")
    st.write({"rows": df.shape[0], "columns": df.shape[1]})

    st.subheader("Columns")
    st.write(list(df.columns))
else:
    st.info("Upload a CSV to begin.")