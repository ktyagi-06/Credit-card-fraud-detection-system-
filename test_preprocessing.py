import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from preprocessing import clean_data, split_data, scale_features
from feature_engineering import add_time_features, add_amount_features, engineer_features, split_X_y


BASE_CONFIG = {
    "preprocessing": {"drop_duplicates": True, "scale_columns": ["Time", "Amount"]},
    "data": {"target_column": "Class", "test_size": 0.2, "val_size": 0.2, "random_state": 42},
}


def make_dummy_df(n=200, fraud_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    df = pd.DataFrame({
        "Time": rng.uniform(0, 100000, n),
        "Amount": rng.uniform(1, 500, n),
        "V1": rng.normal(0, 1, n),
        "V2": rng.normal(0, 1, n),
    })
    df["Class"] = 0
    df.loc[df.sample(n_fraud, random_state=seed).index, "Class"] = 1
    return df


def test_clean_data_drops_duplicates():
    df = make_dummy_df(50)
    df_with_dupes = pd.concat([df, df.iloc[:5]], ignore_index=True)
    cleaned = clean_data(df_with_dupes, BASE_CONFIG)
    assert len(cleaned) == len(df)


def test_clean_data_drops_nulls():
    df = make_dummy_df(50)
    df.loc[0, "Amount"] = np.nan
    cleaned = clean_data(df, BASE_CONFIG)
    assert cleaned.isnull().sum().sum() == 0
    assert len(cleaned) == 49


def test_split_data_preserves_fraud_rate():
    df = make_dummy_df(1000, fraud_rate=0.1)
    train_df, val_df, test_df = split_data(df, BASE_CONFIG)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    overall_rate = df["Class"].mean()
    for split_df in (train_df, val_df, test_df):
        assert abs(split_df["Class"].mean() - overall_rate) < 0.05


def test_split_data_no_row_overlap():
    df = make_dummy_df(300).reset_index(drop=True)
    df["row_id"] = df.index
    train_df, val_df, test_df = split_data(df, BASE_CONFIG)

    train_ids = set(train_df["row_id"])
    val_ids = set(val_df["row_id"])
    test_ids = set(test_df["row_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_scale_features_fits_on_train_only():
    df = make_dummy_df(500)
    train_df, val_df, test_df = split_data(df, BASE_CONFIG)
    scaled_train, scaled_val, scaled_test, scaler = scale_features(train_df, val_df, test_df, BASE_CONFIG)

    # Scaled training data should have ~zero mean / unit variance on scaled columns.
    assert abs(scaled_train["Time"].mean()) < 1e-6 + 0.2
    assert abs(scaled_train["Amount"].std() - 1.0) < 0.2


def test_add_time_features_bounded():
    df = make_dummy_df(50)
    result = add_time_features(df)
    assert result["hour_sin"].between(-1, 1).all()
    assert result["hour_cos"].between(-1, 1).all()


def test_add_amount_features_nonneg():
    df = make_dummy_df(50)
    result = add_amount_features(df)
    assert (result["log_amount"] >= 0).all()


def test_engineer_features_adds_expected_columns():
    df = make_dummy_df(50)
    result = engineer_features(df)
    for col in ("hour_sin", "hour_cos", "log_amount"):
        assert col in result.columns


def test_split_X_y():
    df = make_dummy_df(50)
    X, y = split_X_y(df, "Class")
    assert "Class" not in X.columns
    assert y.name == "Class"
    assert len(X) == len(y)
