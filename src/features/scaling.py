"""Feature scaling utilities for preprocessing."""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Scale feature data using a StandardScaler fitted on the training set.

    The scaler is fit only on `X_train` to avoid data leakage. Numeric
    features that are binary indicators are preserved and not scaled.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.

    Returns:
        A tuple containing the scaled training DataFrame, the scaled test
        DataFrame, and the fitted StandardScaler instance.
    """
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()
    binary_columns = [
        col
        for col in numeric_columns
        if X_train[col].dropna().isin([0, 1]).all()
        and X_train[col].nunique() <= 2
    ]
    scaling_columns = [col for col in numeric_columns if col not in binary_columns]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    if scaling_columns:
        X_train_scaled[scaling_columns] = scaler.fit_transform(
            X_train[scaling_columns]
        )
        X_test_scaled[scaling_columns] = scaler.transform(X_test[scaling_columns])

    return X_train_scaled, X_test_scaled, scaler
