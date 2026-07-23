"""Tests for `scale_features` in `src.features.scaling`.

Verifies that scaling preserves dataframe shape and column names and that
the returned scaler is a `StandardScaler` instance.
"""
from __future__ import annotations

from sklearn.preprocessing import StandardScaler

from src.features.cleaning import convert_total_charges
from src.features.scaling import scale_features


def test_scale_features_preserves_shape_and_columns(sample_features_target):
    """Scaled outputs keep the same shape and columns as the inputs.

    Uses the `sample_features_target` fixture and splits it into a small
    train/test pair to exercise `scale_features` behavior.
    """
    X, _ = sample_features_target

    # Ensure TotalCharges is numeric like real preprocessing does
    X = convert_total_charges(X)

    # Simple train/test split from the small fixture
    X_train = X.iloc[:4].reset_index(drop=True)
    X_test = X.iloc[4:].reset_index(drop=True)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Shapes preserved
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape

    # Column names preserved in the same order
    assert list(X_train_scaled.columns) == list(X_train.columns)
    assert list(X_test_scaled.columns) == list(X_test.columns)

    # Scaler type
    assert isinstance(scaler, StandardScaler)
