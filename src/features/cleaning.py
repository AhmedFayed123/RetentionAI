"""Data cleaning utilities for feature engineering."""

import pandas as pd


def convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the TotalCharges column to numeric values.

    This function attempts to convert the `TotalCharges` column to numeric
    values, forcing invalid parsing to NaN so that later cleaning steps can
    handle missing or malformed entries.

    Args:
        df: Input pandas DataFrame.

    Returns:
        DataFrame with `TotalCharges` converted to numeric dtype.
    """
    df = df.copy()
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def remove_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values from the DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        DataFrame with rows containing any missing values dropped.
    """
    return df.dropna().reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        DataFrame with duplicate rows removed.
    """
    return df.drop_duplicates().reset_index(drop=True)


def reset_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    """Reset the index of the DataFrame.

    Args:
        df: Input pandas DataFrame.

    Returns:
        DataFrame with a reset default integer index.
    """
    return df.reset_index(drop=True)
