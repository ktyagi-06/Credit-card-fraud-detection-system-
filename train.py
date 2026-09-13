"""
Train all models enabled in config.yaml and save them to the artifacts
directory, along with the fitted scaler and the held-out val/test splits
(so evaluate.py can reuse the exact same splits).

Usage:
    python src/train.py --config config.yaml
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loader import load_config, load_raw_data
from preprocessing import clean_data, split_data, scale_features
from feature_engineering import engineer_features, split_X_y, resample_training_data


def build_neural_net(input_dim: int, params: dict):
    """Built lazily so the rest of the pipeline works even if TensorFlow
    isn't installed / the neural_net model is disabled in config."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for units in params["hidden_layers"]:
        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.Dropout(params["dropout"]))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return model


def train_all_models(config: dict):
    model_dir = Path(config["artifacts"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning data...")
    df = load_raw_data(config)
    df = clean_data(df, config)

    print("\nSplitting data...")
    train_df, val_df, test_df = split_data(df, config)

    print("\nScaling Time/Amount...")
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df, config)
    joblib.dump(scaler, model_dir / "scaler.pkl")

    print("\nEngineering features...")
    train_df = engineer_features(train_df)
    val_df = engineer_features(val_df)
    test_df = engineer_features(test_df)

    # Persist splits so evaluate.py/predict.py use exactly the same data.
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    target_col = config["data"]["target_column"]
    X_train, y_train = split_X_y(train_df, target_col)

    print("\nResampling training data...")
    X_train_res, y_train_res = resample_training_data(X_train, y_train, config)

    # Save feature column order for consistent inference later.
    joblib.dump(list(X_train.columns), model_dir / "feature_columns.pkl")

    use_class_weight = config["resampling"]["method"] == "class_weight"

    models_cfg = config["models"]

    if models_cfg["logistic_regression"]["enabled"]:
        print("\nTraining Logistic Regression...")
        params = dict(models_cfg["logistic_regression"]["params"])
        if not use_class_weight:
            params.pop("class_weight", None)
        clf = LogisticRegression(**params)
        clf.fit(X_train_res, y_train_res)
        joblib.dump(clf, model_dir / "logistic_regression_model.pkl")

    if models_cfg["random_forest"]["enabled"]:
        print("Training Random Forest...")
        params = dict(models_cfg["random_forest"]["params"])
        if not use_class_weight:
            params.pop("class_weight", None)
        clf = RandomForestClassifier(**params)
        clf.fit(X_train_res, y_train_res)
        joblib.dump(clf, model_dir / "random_forest_model.pkl")

    if models_cfg["xgboost"]["enabled"]:
        print("Training XGBoost...")
        params = dict(models_cfg["xgboost"]["params"])
        if use_class_weight:
            neg, pos = np.bincount(y_train_res)
            params["scale_pos_weight"] = neg / max(pos, 1)
        clf = XGBClassifier(**params)
        clf.fit(X_train_res, y_train_res)
        joblib.dump(clf, model_dir / "xgboost_model.pkl")

    if models_cfg["neural_net"]["enabled"]:
        print("Training Neural Net...")
        params = models_cfg["neural_net"]["params"]
        nn = build_neural_net(X_train_res.shape[1], params)
        class_weight = None
        if use_class_weight:
            neg, pos = np.bincount(y_train_res)
            total = neg + pos
            class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}
        nn.fit(
            X_train_res, y_train_res,
            epochs=params["epochs"], batch_size=params["batch_size"],
            class_weight=class_weight, verbose=2,
        )
        nn.save(model_dir / "neural_net_model.keras")

    print(f"\nAll enabled models saved to {model_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    train_all_models(config)


if __name__ == "__main__":
    main()
