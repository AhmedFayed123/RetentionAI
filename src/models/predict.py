from typing import Any, Union

import numpy as np
import pandas as pd


def predict(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Return predictions from a fitted model.

    Parameters
    ----------
    model : Any
        A fitted estimator implementing ``predict``.
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix to predict on.

    Returns
    -------
    numpy.ndarray
        Predicted labels as a 1-D numpy array.
    """
    preds = model.predict(X)
    return np.asarray(preds)


def predict_probabilities(model: Any, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Return the positive-class probabilities from a fitted model.

    Parameters
    ----------
    model : Any
        A fitted estimator implementing ``predict_proba`` and preferably
        exposing a ``classes_`` attribute for class ordering.
    X : pandas.DataFrame or numpy.ndarray
        Feature matrix to predict probabilities on.

    Returns
    -------
    numpy.ndarray
        1-D array with probability of the positive class for each sample.

    Raises
    ------
    AttributeError
        If the model does not implement ``predict_proba``.
    ValueError
        If the positive class cannot be determined from ``model.classes_``
        and the probability matrix shape is ambiguous.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError(
            "The provided model does not support probability estimates (no `predict_proba`)."
        )

    probs = model.predict_proba(X)
    probs = np.asarray(probs)

    # Prefer using model.classes_ to locate the positive class (1)
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        try:
            pos_idx = classes.index(1)
        except ValueError:
            if len(classes) == 2:
                # fallback: assume the second column is the positive class
                pos_idx = 1
            else:
                raise ValueError(
                    f"Unable to determine positive class from model.classes_={classes}."
                )
    else:
        # No classes_ attribute: for binary case assume positive is column 1
        if probs.ndim == 2 and probs.shape[1] == 2:
            pos_idx = 1
        else:
            raise ValueError(
                "Model has no `classes_` attribute and returned probability matrix with shape "
                f"{probs.shape}; cannot infer which column is the positive class."
            )

    return probs[:, pos_idx]
