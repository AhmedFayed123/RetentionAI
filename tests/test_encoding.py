"""Tests for encoding utilities in `src.features.encoding`.

Verifies that the target is encoded to integers and that one-hot
encoding transforms categorical columns while preserving numeric columns.
"""
from __future__ import annotations

from typing import List

import pandas as pd
from pandas.testing import assert_series_equal

from src.features.cleaning import convert_total_charges
from src.features.encoding import encode_target, one_hot_encode


def test_encode_target_converts_churn_to_int(sample_dataframe: pd.DataFrame) -> None:
    """`encode_target` converts 'Yes'/'No' to 1/0 integers."""
    df = sample_dataframe.copy()

    # The fixture may use pandas' nullable StringDtype; check for string-like dtype
    assert pd.api.types.is_string_dtype(df["Churn"]) or pd.api.types.is_object_dtype(df["Churn"])

    encoded = encode_target(df)

    # Expect Churn column present and numeric 0/1 values
    assert "Churn" in encoded.columns
    assert pd.api.types.is_integer_dtype(encoded["Churn"]) or pd.api.types.is_integer_dtype(
        encoded["Churn"].dropna()
    )

    # Explicit expected mapping for the fixture
    expected = pd.Series([0, 1, 0, 0, 1, 0], name="Churn")
    assert_series_equal(encoded["Churn"].reset_index(drop=True), expected)


def test_one_hot_encode_preserves_numeric_and_encodes_categoricals(sample_dataframe: pd.DataFrame) -> None:
    """`one_hot_encode` should one-hot encode categorical columns and keep numeric columns.

    We convert `TotalCharges` first so it is numeric (mirrors real preprocessing).
    """
    df = sample_dataframe.copy()
    df = convert_total_charges(df)

    numeric_cols: List[str] = ["SeniorCitizen", "MonthlyCharges", "TotalCharges"]

    encoded = one_hot_encode(df)

    # Numeric columns should still be present
    for col in numeric_cols:
        assert col in encoded.columns

    # Original categorical columns should be replaced by dummy columns
    assert "Gender" not in encoded.columns
    assert any(c.startswith("Gender_") for c in encoded.columns)

    # Churn should be encoded into a dummy column (e.g. 'Churn_Yes')
    assert "Churn" not in encoded.columns
    assert any(c.startswith("Churn_") for c in encoded.columns)

    # Verify numeric values were preserved
    pd.testing.assert_series_equal(encoded["SeniorCitizen"].reset_index(drop=True), df["SeniorCitizen"].reset_index(drop=True))
