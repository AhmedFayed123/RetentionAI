"""Tests for `calculate_metrics` in `src.evaluation.metrics`.

Verifies that the returned mapping contains expected keys and that each
metric value lies between 0 and 1 (inclusive) for typical inputs.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from src.evaluation.metrics import calculate_metrics


def _assert_metrics_in_range(metrics: dict[str, float]) -> None:
    """Helper to assert every metric is between 0 and 1 inclusive."""
    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics.keys()) == expected_keys

    for k, v in metrics.items():
        # roc_auc may sometimes be NaN for degenerate inputs; ensure finite
        # values are within [0, 1].
        if np.isnan(v):
            continue
        assert 0.0 <= v <= 1.0, f"Metric {k}={v} out of range"


def test_calculate_metrics_perfect_predictions(sample_features_target: Iterable) -> None:
    """When predictions exactly match the truth, most metrics should be 1.0."""
    _, y = sample_features_target

    y_true = np.asarray(y)
    y_pred = y_true.copy()

    metrics = calculate_metrics(y_true, y_pred)

    _assert_metrics_in_range(metrics)

    # Perfect predictions: accuracy, precision, recall, f1 should be 1.0.
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    # roc_auc should also be 1.0 for non-degenerate label sets
    assert metrics["roc_auc"] == 1.0


def test_calculate_metrics_imperfect_predictions(sample_features_target: Iterable) -> None:
    """Verify metrics remain in range for imperfect predictions."""
    _, y = sample_features_target
    y_true = np.asarray(y)

    # Create a contrived imperfect prediction (flip first label)
    y_pred = y_true.copy()
    if y_pred.size > 0:
        y_pred[0] = 1 - int(y_pred[0])

    metrics = calculate_metrics(y_true, y_pred)

    _assert_metrics_in_range(metrics)
