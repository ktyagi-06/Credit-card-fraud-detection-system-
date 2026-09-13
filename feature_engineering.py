"""
Feature engineering and resampling.

Adds a couple of lightweight derived features on top of the raw
Time/Amount/PCA columns, then applies the configured resampling
strategy (SMOTE or class weighting) to the TRAINING split only.
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day-like cyclical features from the raw 'Time' column
    (seconds elapsed since the first transaction in the dataset)."""
    df = df.copy()
    seconds_in_day = 24 * 60 * 60
    time_of_day = df["Time"] % seconds_in_day
    df["hour_sin"] = np.sin(2 * np.pi * time_of_day / seconds_in_day)
    df["hour_cos"] = np.cos(2 * np.pi * time_of_day / seconds_in_day)
    return df


def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"].clip(lower=0))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_amount_features(df)
    return df


def split_X_y(df: pd.DataFrame, target_col: str):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def resample_training_data(X_train: pd.DataFrame, y_train: pd.Series, config: dict):
    """Apply the configured resampling strategy. Only ever call this on the
    training split -- resampling val/test would leak information and give
    an unrealistically optimistic evaluation."""
    method = config["resampling"]["method"]

    if method == "smote":
        sampling_strategy = config["resampling"]["smote_sampling_strategy"]
        random_state = config["resampling"]["random_state"]
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"SMOTE: {len(X_train):,} -> {len(X_resampled):,} rows "
              f"(fraud rate {y_train.mean()*100:.3f}% -> {y_resampled.mean()*100:.3f}%)")
        return X_resampled, y_resampled

    if method in ("class_weight", "none"):
        return X_train, y_train

    raise ValueError(f"Unknown resampling method: {method}")
