"""
Config loading and raw data loading utilities.
"""
from pathlib import Path
import yaml
import pandas as pd


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_raw_data(config: dict) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No data found at {raw_path}. Either place the Kaggle "
            f"creditcard.csv there, or run "
            f"`python src/generate_synthetic_data.py --output {raw_path}` "
            f"to create a synthetic dataset with the same schema."
        )
    df = pd.read_csv(raw_path)

    target_col = config["data"]["target_column"]
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found in {raw_path}.")

    return df
