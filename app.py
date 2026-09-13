"""
Minimal Streamlit demo: upload a CSV of transactions (or edit a single
transaction manually) and see fraud probabilities from a trained model.

Run with:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.feature_engineering import engineer_features

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")
st.title("Credit Card Fraud Detection — Demo")

MODEL_DIR = Path("models")
AVAILABLE_MODELS = {
    "XGBoost": MODEL_DIR / "xgboost_model.pkl",
    "Random Forest": MODEL_DIR / "random_forest_model.pkl",
    "Logistic Regression": MODEL_DIR / "logistic_regression_model.pkl",
}
AVAILABLE_MODELS = {k: v for k, v in AVAILABLE_MODELS.items() if v.exists()}

if not AVAILABLE_MODELS:
    st.warning("No trained models found in `models/`. Run `python src/train.py` first.")
    st.stop()

model_choice = st.sidebar.selectbox("Model", list(AVAILABLE_MODELS.keys()))
threshold = st.sidebar.slider("Fraud probability threshold", 0.0, 1.0, 0.5, 0.01)

scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")
model = joblib.load(AVAILABLE_MODELS[model_choice])

uploaded = st.file_uploader("Upload a CSV of transactions (Time, V1..V28, Amount columns)", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    raw_df = df.copy()
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    df = engineer_features(df)
    df = df[feature_columns]

    probs = model.predict_proba(df)[:, 1]
    raw_df["fraud_probability"] = probs
    raw_df["flagged"] = (probs >= threshold).astype(int)

    st.subheader(f"Results — {raw_df['flagged'].sum()} of {len(raw_df)} flagged")
    st.dataframe(raw_df.sort_values("fraud_probability", ascending=False))

    st.download_button(
        "Download scored CSV",
        raw_df.to_csv(index=False).encode("utf-8"),
        file_name="scored_transactions.csv",
        mime="text/csv",
    )
else:
    st.info("Upload a CSV to score transactions, or use `src/predict.py` for single-transaction scoring from the command line.")
