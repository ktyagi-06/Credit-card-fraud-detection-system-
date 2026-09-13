"""
Evaluate all trained models on the held-out test split, using metrics
appropriate for severe class imbalance (PR-AUC as primary, plus ROC-AUC,
F1, and recall achievable at a fixed precision target). Also sweeps
classification thresholds and saves a precision/recall-vs-threshold plot.

Usage:
    python src/evaluate.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, classification_report,
)

from data_loader import load_config


def load_model(model_dir: Path, name: str):
    path_pkl = model_dir / f"{name}_model.pkl"
    path_keras = model_dir / f"{name}_model.keras"
    if path_pkl.exists():
        return joblib.load(path_pkl), "sklearn"
    if path_keras.exists():
        from tensorflow import keras
        return keras.models.load_model(path_keras), "keras"
    return None, None


def get_probabilities(model, kind: str, X: pd.DataFrame) -> np.ndarray:
    if kind == "sklearn":
        return model.predict_proba(X)[:, 1]
    if kind == "keras":
        return model.predict(X, verbose=0).ravel()
    raise ValueError(f"Unknown model kind: {kind}")


def recall_at_precision(y_true, y_scores, target_precision: float):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # precision/recall arrays are one element longer than thresholds
    valid = precision[:-1] >= target_precision
    if not valid.any():
        return 0.0, None
    idx = np.argmax(valid)  # first threshold hitting the target precision
    return recall[idx], thresholds[idx]


def evaluate_model(name: str, y_true, y_scores, target_precision: float) -> dict:
    pr_auc = average_precision_score(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    y_pred_default = (y_scores >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_default)
    recall_at_p, threshold_at_p = recall_at_precision(y_true, y_scores, target_precision)
    cm = confusion_matrix(y_true, y_pred_default).tolist()

    print(f"\n=== {name} ===")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1 (0.5 threshold): {f1:.4f}")
    print(f"Recall @ {target_precision:.0%} precision: {recall_at_p:.4f} "
          f"(threshold={threshold_at_p if threshold_at_p is not None else 'n/a'})")
    print("Confusion matrix (0.5 threshold):")
    print(pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]))
    print(classification_report(y_true, y_pred_default, digits=4))

    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_at_0.5": f1,
        f"recall_at_{target_precision}_precision": recall_at_p,
        "confusion_matrix_at_0.5": cm,
    }


def plot_pr_curves(results_scores: dict, y_true, output_path: Path):
    plt.figure(figsize=(7, 6))
    for name, scores in results_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved PR curve plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained fraud detection models.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    model_dir = Path(config["artifacts"]["model_dir"])
    processed_dir = Path(config["data"]["processed_dir"])
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_parquet(processed_dir / "test.parquet")
    target_col = config["data"]["target_column"]
    feature_columns = joblib.load(model_dir / "feature_columns.pkl")

    X_test = test_df[feature_columns]
    y_test = test_df[target_col]

    target_precision = config["evaluation"]["target_precision_for_recall_report"]

    model_names = ["logistic_regression", "random_forest", "xgboost", "neural_net"]
    all_results = []
    all_scores = {}

    for name in model_names:
        model, kind = load_model(model_dir, name)
        if model is None:
            continue
        scores = get_probabilities(model, kind, X_test)
        all_scores[name] = scores
        all_results.append(evaluate_model(name, y_test, scores, target_precision))

    if not all_results:
        print("No trained models found. Run src/train.py first.")
        return

    results_df = pd.DataFrame(all_results).sort_values("pr_auc", ascending=False)
    results_df.to_csv(output_dir / "model_comparison.csv", index=False)
    with open(output_dir / "model_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)

    plot_pr_curves(all_scores, y_test, output_dir / "pr_curves.png")

    print("\n=== Summary (sorted by PR-AUC) ===")
    print(results_df[["model", "pr_auc", "roc_auc", "f1_at_0.5"]].to_string(index=False))
    print(f"\nFull report saved to {output_dir}/")


if __name__ == "__main__":
    main()
