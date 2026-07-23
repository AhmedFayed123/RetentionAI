from typing import Mapping

import pandas as pd


def append_model_results(
    comparison_df: pd.DataFrame, model_name: str, metrics: Mapping[str, float]
) -> pd.DataFrame:
    """Append model metrics to a comparison DataFrame.

    If `comparison_df` is empty or None, a new DataFrame with the expected
    columns will be created.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Existing comparison DataFrame. May be empty or None.
    model_name : str
        Human-readable model name to append.
    metrics : Mapping[str, float]
        Dictionary produced by ``calculate_metrics`` containing keys
        ``accuracy``, ``precision``, ``recall``, ``f1``, and ``roc_auc``.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with the appended row.
    """
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]

    # Prepare row values with safe defaults
    row = {
        "model": model_name,
        "accuracy": float(metrics.get("accuracy", float("nan"))),
        "precision": float(metrics.get("precision", float("nan"))),
        "recall": float(metrics.get("recall", float("nan"))),
        "f1": float(metrics.get("f1", float("nan"))),
        "roc_auc": float(metrics.get("roc_auc", float("nan"))),
    }

    new_row_df = pd.DataFrame([row], columns=cols)

    if comparison_df is None or comparison_df.empty:
        return new_row_df

    # Ensure comparison_df has the required columns
    for c in cols:
        if c not in comparison_df.columns:
            comparison_df[c] = pd.NA

    updated = pd.concat([comparison_df, new_row_df], ignore_index=True, sort=False)
    return updated


def sort_models(comparison_df: pd.DataFrame, metric: str = "f1") -> pd.DataFrame:
    """Return a copy of `comparison_df` sorted descending by `metric`.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame containing model comparison rows. Must include `metric`.
    metric : str, optional
        Metric column name to sort by (default: "f1").

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame (descending by `metric`). If `metric` is not present
        in `comparison_df`, the original DataFrame is returned unchanged.
    """
    if metric not in comparison_df.columns:
        return comparison_df.copy()

    return comparison_df.sort_values(by=metric, ascending=False).reset_index(drop=True)
