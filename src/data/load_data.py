"""Utilities for loading project datasets."""

from pathlib import Path

import pandas as pd


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load a dataset from a CSV file path.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the loaded data.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    return pd.read_csv(path)
