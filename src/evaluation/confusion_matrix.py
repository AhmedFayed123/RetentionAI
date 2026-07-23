from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> None:
    """Plot the confusion matrix for a fitted classification model.

    This function uses `ConfusionMatrixDisplay.from_estimator` to compute and
    display the confusion matrix for `model` on `X_test` / `y_test`.

    Parameters
    ----------
    model : Any
        A fitted classification estimator implementing ``predict`` (and
        optionally ``predict_proba``).
    X_test : pandas.DataFrame or numpy.ndarray
        Test feature matrix.
    y_test : pandas.Series or numpy.ndarray
        True labels for the test set.

    Returns
    -------
    None
        Displays the confusion matrix figure and returns nothing.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
