"""Utilities for loading project datasets."""

import pandas as pd

from src.config.settings import RAW_DATA_PATH


def load_dataset() -> pd.DataFrame:
    """Load the configured raw dataset from CSV format.

    Returns:
        The raw dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the configured raw dataset file does not exist.
    """
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Raw data file not found: {RAW_DATA_PATH}")

    return pd.read_csv(RAW_DATA_PATH)
