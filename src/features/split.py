"""Dataset splitting utilities for preprocessing."""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into train and test sets.

    The function uses stratified sampling on the target variable to preserve
    class distribution across training and test sets.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of the dataset to reserve for testing.
        random_state: Random seed for reproducibility.

    Returns:
        A tuple containing X_train, X_test, y_train, and y_test.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
