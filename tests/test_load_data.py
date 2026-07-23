"""Tests for `src.data.load_data.load_dataset`.

Includes:
- valid CSV path returns a DataFrame
- invalid path raises FileNotFoundError
"""
from pathlib import Path

import pandas as pd
import pytest

from src.data.load_data import load_dataset


def test_load_dataset_valid_path(tmp_path: Path) -> None:
    """Given a valid CSV file path, load_dataset returns a DataFrame.

    Uses `tmp_path` to create a temporary CSV file with a few rows and
    verifies the resulting DataFrame shape and columns.
    """
    data = {
        "Gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "MonthlyCharges": [29.85, 56.95],
        "TotalCharges": ["29.85", "1889.5"],
        "Churn": ["No", "Yes"],
    }
    df = pd.DataFrame(data)

    file_path = tmp_path / "sample.csv"
    df.to_csv(file_path, index=False)

    loaded = load_dataset(str(file_path))

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == df.shape
    assert list(loaded.columns) == list(df.columns)


def test_load_dataset_invalid_path_raises() -> None:
    """When the CSV path does not exist, a FileNotFoundError is raised."""
    missing = Path("nonexistent_file_hopefully_missing.csv")

    with pytest.raises(FileNotFoundError):
        load_dataset(str(missing))
