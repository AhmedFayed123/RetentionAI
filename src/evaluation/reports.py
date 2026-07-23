from typing import Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report as sklearn_classification_report


def generate_classification_report(
    y_true: Union[Sequence[int], np.ndarray, pd.Series],
    y_pred: Union[Sequence[int], np.ndarray, pd.Series],
) -> str:
    """Generate a scikit-learn classification report.

    Parameters
    ----------
    y_true : Sequence[int] | numpy.ndarray | pandas.Series
        True binary labels (0/1).
    y_pred : Sequence[int] | numpy.ndarray | pandas.Series
        Predicted labels.

    Returns
    -------
    str
        The textual classification report produced by
        ``sklearn.metrics.classification_report``.
    """
    return sklearn_classification_report(y_true, y_pred)
