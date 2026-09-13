"""
Run inference on a single new transaction using a trained model.

Usage:
    python src/predict.py --model models/xgboost_model.pkl --input sample_transaction.json

The input JSON should contain raw fields: Time, Amount, V1..V28
(the script applies the same scaler + feature engineering used in training).
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from feature_engineering import engineer_features


def predict_single(model_path: str, input_path: str, model_dir: str = "models", threshold: float = 0.5):
    model_dir = Path(model_dir)
    scaler = joblib.load(model_dir / "scaler.pkl")
    feature_columns = joblib.load(model_dir / "feature_columns.pkl")

    with open(input_path, "r") as f:
        transaction = json.load(f)

    df = pd.DataFrame([transaction])
    df[["Time", "Amount"]] = scaler.transform(df[["Time", "Amount"]])
    df = engineer_features(df)
    df = df[feature_columns]

    model_path = Path(model_path)
    if model_path.suffix == ".keras":
        from tensorflow import keras
        model = keras.models.load_model(model_path)
        prob = float(model.predict(df, verbose=0).ravel()[0])
    else:
        model = joblib.load(model_path)
        prob = float(model.predict_proba(df)[:, 1][0])

    prediction = int(prob >= threshold)
    return {"fraud_probability": prob, "prediction": prediction, "threshold_used": threshold}


def main():
    parser = argparse.ArgumentParser(description="Score a single transaction for fraud.")
    parser.add_argument("--model", type=str, required=True, help="Path to a saved model file.")
    parser.add_argument("--input", type=str, required=True, help="Path to a JSON file with transaction fields.")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with scaler.pkl / feature_columns.pkl.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    result = predict_single(args.model, args.input, args.model_dir, args.threshold)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
