"""Tests for cleaning utilities in `src.features.cleaning`.

Uses the `sample_dataframe` fixture from `tests/conftest.py` to construct
realistic test cases covering missing values, duplicates, and index reset
behaviour.
"""
from __future__ import annotations

import pandas as pd

from src.features.cleaning import (
    convert_total_charges,
    remove_duplicates,
    remove_missing_values,
    reset_dataframe_index,
)


def test_remove_missing_values(sample_dataframe: pd.DataFrame) -> None:
    """`remove_missing_values` drops rows with NaN after conversion.

    We first convert `TotalCharges` to numeric so empty strings become NaN,
    then ensure rows with missing values are removed and no NaNs remain.
    """
    df = convert_total_charges(sample_dataframe)

    # Confirm there is at least one NaN introduced by conversion
    assert df["TotalCharges"].isna().any()

    cleaned = remove_missing_values(df)

    # No missing values remain
    assert cleaned.isna().sum().sum() == 0

    # Shape equals the original dropped-NaN shape
    expected_rows = df.dropna().shape[0]
    assert cleaned.shape[0] == expected_rows


def test_remove_duplicates(sample_dataframe: pd.DataFrame) -> None:
    """`remove_duplicates` removes duplicate rows and resets the index."""
    df = sample_dataframe.copy()

    # Introduce an explicit duplicate of the first row
    duplicate_row = df.iloc[[0]].copy()
    df_dup = pd.concat([df, duplicate_row], ignore_index=True)

    assert df_dup.duplicated().any()

    deduped = remove_duplicates(df_dup)

    # No duplicates remain
    assert not deduped.duplicated().any()

    # Number of rows equals number of unique rows in the original
    assert deduped.shape[0] == df.drop_duplicates().shape[0]


def test_reset_dataframe_index(sample_dataframe: pd.DataFrame) -> None:
    """`reset_dataframe_index` returns a DataFrame with a contiguous RangeIndex."""
    df = sample_dataframe.copy()

    # Drop the first row to create a non-zero-based index
    df_dropped = df.drop(index=0)
    assert list(df_dropped.index)[0] != 0

    reset = reset_dataframe_index(df_dropped)

    # Index should be a RangeIndex starting at 0
    assert isinstance(reset.index, pd.RangeIndex)
    assert reset.index.start == 0
    assert reset.index.stop == reset.shape[0]
