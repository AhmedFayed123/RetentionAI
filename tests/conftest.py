"""Pytest fixtures for RetentionAI tests.

Provides reusable sample data that resembles the customer churn dataset
used throughout the project. Fixtures here are intentionally small and
easy to inspect, but include realistic value distributions and a missing
`TotalCharges` entry to mimic real-world dirty data.
"""
from __future__ import annotations

from typing import Tuple

import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Return a small sample DataFrame similar to the churn dataset.

    Columns included:
    - Gender: categorical string values ('Male'/'Female')
    - SeniorCitizen: binary indicator (0/1)
    - MonthlyCharges: continuous monthly charge (float)
    - TotalCharges: total charges (string or float) to mimic raw CSV
    - Churn: target column with 'Yes'/'No'
    """
    data = {
        "Gender": ["Female", "Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1, 0],
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65],
        # Mix numeric-as-string and an empty string to simulate dirty CSVs
        "TotalCharges": ["29.85", "1889.5", "108.15", "1840.75", "151.65", ""],
        "Churn": ["No", "Yes", "No", "No", "Yes", "No"],
    }

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_features_target(sample_dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target vector y from the sample DataFrame.

    - Encodes `Churn` to integers: 'No' -> 0, 'Yes' -> 1.
    - Returns X (DataFrame) and y (Series) so tests can reuse them directly.
    """
    df = sample_dataframe.copy()

    # Map churn to numeric labels (0/1). This mirrors the encoding step
    # typically performed in the preprocessing module.
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Features are all columns except the target
    X = df.drop(columns=["Churn"]).reset_index(drop=True)
    y = y.reset_index(drop=True)

    return X, y
