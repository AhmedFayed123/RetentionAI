from typing import Any, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> Mapping[str, float]:
    """Calculate common classification metrics.

    Parameters
    ----------
    y_true : Sequence or numpy.ndarray or pandas.Series
        True binary labels (0/1).
    y_pred : Sequence or numpy.ndarray or pandas.Series
        Predicted labels or scores. For ``roc_auc`` a score/probability
        for the positive class is preferred; if only class labels are
        provided, ``roc_auc`` will be attempted and may return ``nan`` if
        not computable.

    Returns
    -------
    Mapping[str, float]
        Dictionary with keys ``accuracy``, ``precision``, ``recall``,
        ``f1`` and ``roc_auc`` containing the corresponding metric values.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision = precision_score(y_true_arr, y_pred_arr, zero_division=0)
    recall = recall_score(y_true_arr, y_pred_arr, zero_division=0)
    f1 = f1_score(y_true_arr, y_pred_arr, zero_division=0)

    # Compute ROC AUC when possible; fall back to NaN on failure.
    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_pred_arr))
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }
