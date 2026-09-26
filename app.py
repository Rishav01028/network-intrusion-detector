"""
app.py
Streamlit dashboard: upload a CSV of traffic records, get each one flagged
as normal or a specific attack category, using the model trained by
train_model.py.

Run with:
    streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

from train_model import NUMERIC_FEATURES, CATEGORICAL_FEATURES

st.set_page_config(page_title="Network Intrusion Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ Network Intrusion Detection System")
st.caption(
    "Upload a CSV of network traffic records and this dashboard will flag each "
    "one as normal or as a specific attack type (DoS, Probe, R2L, U2R), using a "
    "Random Forest model trained with 5-fold cross-validation."
)

REQUIRED_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_resource
def load_model():
    return joblib.load("model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error("No trained model found. Run `python train_model.py` first to create model.joblib.")
    st.stop()

st.subheader("1. Upload traffic log")
st.caption(f"CSV must contain these columns: {', '.join(REQUIRED_COLS)}")

uploaded = st.file_uploader("Traffic CSV", type=["csv"])

demo = st.checkbox("...or use a small built-in demo sample instead", value=uploaded is None)

if demo and uploaded is None:
    df = pd.read_csv("data/traffic_dataset.csv").sample(15, random_state=7).reset_index(drop=True)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
elif uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = None

if df is not None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        st.subheader("2. Results")
        preds = model.predict(df[REQUIRED_COLS])
        proba = model.predict_proba(df[REQUIRED_COLS])
        confidence = proba.max(axis=1)

        results = df.copy()
        results["prediction"] = preds
        results["confidence"] = (confidence * 100).round(1).astype(str) + "%"

        def highlight(row):
            color = "" if row["prediction"] == "normal" else "background-color: #ffe3e3"
            return [color] * len(row)

        st.dataframe(results.style.apply(highlight, axis=1), use_container_width=True)

        n_flagged = (results["prediction"] != "normal").sum()
        if n_flagged:
            st.warning(f"⚠️ {n_flagged} of {len(results)} records flagged as potentially malicious.")
        else:
            st.success("No suspicious traffic detected in this batch.")

        st.subheader("3. Breakdown by predicted class")
        st.bar_chart(results["prediction"].value_counts())
