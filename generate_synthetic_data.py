"""
Generate a synthetic dataset matching the schema of the Kaggle
"Credit Card Fraud Detection" dataset: Time, V1..V28, Amount, Class.

This lets the rest of the pipeline run end-to-end without requiring
a Kaggle download. It is NOT a substitute for the real dataset --
it approximates the statistical shape (heavy class imbalance,
roughly Gaussian PCA-like features, right-skewed transaction
amounts) but real fraud patterns are more subtle than this.

Usage:
    python src/generate_synthetic_data.py --n-rows 100000 --fraud-rate 0.0017
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_transactions(n_rows: int, fraud_rate: float, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    n_fraud = max(1, int(n_rows * fraud_rate))
    n_normal = n_rows - n_fraud

    n_pca_features = 28

    # Normal transactions: PCA-like features centered near 0 with modest spread
    normal_pca = rng.normal(loc=0.0, scale=1.0, size=(n_normal, n_pca_features))
    # Fraudulent transactions: shifted mean and higher variance on a subset of
    # features, mimicking the fact that fraud tends to look "different" in
    # a handful of the anonymized dimensions rather than uniformly.
    fraud_pca = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, n_pca_features))
    anomalous_dims = rng.choice(n_pca_features, size=6, replace=False)
    fraud_pca[:, anomalous_dims] += rng.normal(loc=3.5, scale=1.5, size=(n_fraud, len(anomalous_dims)))

    normal_time = rng.uniform(0, 172800, size=n_normal)   # 2-day window in seconds
    fraud_time = rng.uniform(0, 172800, size=n_fraud)

    # Amounts: right-skewed (lognormal), fraud tends to cluster at smaller
    # "test" amounts and occasional large amounts -- a simplification, but
    # a reasonable synthetic approximation.
    normal_amount = rng.lognormal(mean=3.0, sigma=1.2, size=n_normal)
    fraud_amount = np.concatenate([
        rng.lognormal(mean=1.5, sigma=1.0, size=n_fraud // 2),
        rng.lognormal(mean=4.5, sigma=1.0, size=n_fraud - n_fraud // 2),
    ])
    rng.shuffle(fraud_amount)

    normal_df = pd.DataFrame(normal_pca, columns=[f"V{i}" for i in range(1, n_pca_features + 1)])
    normal_df["Time"] = normal_time
    normal_df["Amount"] = np.round(normal_amount, 2)
    normal_df["Class"] = 0

    fraud_df = pd.DataFrame(fraud_pca, columns=[f"V{i}" for i in range(1, n_pca_features + 1)])
    fraud_df["Time"] = fraud_time
    fraud_df["Amount"] = np.round(fraud_amount, 2)
    fraud_df["Class"] = 1

    df = pd.concat([normal_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Reorder columns to match the real dataset: Time, V1..V28, Amount, Class
    cols = ["Time"] + [f"V{i}" for i in range(1, n_pca_features + 1)] + ["Amount", "Class"]
    return df[cols]


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic credit card transaction data.")
    parser.add_argument("--n-rows", type=int, default=100_000, help="Total number of transactions to generate.")
    parser.add_argument("--fraud-rate", type=float, default=0.0017, help="Fraction of transactions that are fraud.")
    parser.add_argument("--output", type=str, default="data/creditcard.csv", help="Output CSV path.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = generate_synthetic_transactions(args.n_rows, args.fraud_rate, args.random_state)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df):,} rows ({df['Class'].sum():,} fraud, "
          f"{df['Class'].mean()*100:.3f}% fraud rate) -> {output_path}")


if __name__ == "__main__":
    main()
