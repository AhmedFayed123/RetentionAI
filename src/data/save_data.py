"""Utilities for saving datasets to disk."""

from pathlib import Path

import pandas as pd


def save_dataframe(df: pd.DataFrame, output_path: str) -> None:
    """Save a pandas DataFrame to CSV format.

    This function creates any missing parent directories before saving the
    dataframe as a CSV file with `index=False`.

    Args:
        df: DataFrame to save.
        output_path: Path where the CSV file should be written.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved dataframe to: {path}")
