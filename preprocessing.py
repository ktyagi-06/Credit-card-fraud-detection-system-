"""
Preprocessing: deduplication, scaling of Time/Amount, and stratified
train/val/test splitting. PCA features (V1..V28) are left as-is since
they're already roughly standardized in the source dataset.
"""
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if config["preprocessing"].get("drop_duplicates", True):
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            print(f"Dropped {removed:,} duplicate rows.")

    if df.isnull().any().any():
        na_counts = df.isnull().sum()
        print(f"Warning: found missing values, dropping affected rows:\n{na_counts[na_counts > 0]}")
        df = df.dropna()

    return df.reset_index(drop=True)


def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                    config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit a scaler on train only, apply to val/test to avoid leakage."""
    cols = config["preprocessing"]["scale_columns"]
    scaler = StandardScaler()

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df[cols] = scaler.fit_transform(train_df[cols])
    val_df[cols] = scaler.transform(val_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])

    return train_df, val_df, test_df, scaler


def split_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train/val/test, preserving the fraud rate in each split."""
    target_col = config["data"]["target_column"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    random_state = config["data"]["random_state"]

    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target_col], random_state=random_state
    )
    # val_size is expressed as a fraction of the ORIGINAL data; rescale
    # relative to what remains after removing the test split.
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val_size, stratify=train_val_df[target_col],
        random_state=random_state
    )

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        rate = split_df[target_col].mean() * 100
        print(f"{name}: {len(split_df):,} rows, {rate:.4f}% fraud")

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
