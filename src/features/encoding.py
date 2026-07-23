"""Feature encoding utilities for preprocessing."""

import pandas as pd


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the churn target column as numeric values.

    This function converts the `Churn` column values from `Yes`/`No` to
    `1`/`0`. The input DataFrame is copied before modification to preserve
    the original data.

    Args:
        df: Input pandas DataFrame.

    Returns:
        DataFrame with the `Churn` column encoded as integers.
    """
    df = df.copy()
    if "Churn" in df.columns:
        s = df["Churn"]
        # Already numeric 0/1 -> keep as ints
        try:
            non_null_vals = set(s.dropna().unique())
        except Exception:
            non_null_vals = set()

        if pd.api.types.is_numeric_dtype(s) and non_null_vals.issubset({0, 1}):
            df["Churn"] = s.astype(int)
        # Typical string values 'Yes'/'No'
        elif non_null_vals.issubset({"Yes", "No"}):
            df["Churn"] = s.map({"No": 0, "Yes": 1})
        else:
            # Best-effort: try mapping then numeric coercion
            mapped = s.map({"No": 0, "Yes": 1})
            if mapped.isna().sum() == 0:
                df["Churn"] = mapped
            else:
                coerced = pd.to_numeric(s, errors="coerce")
                if pd.api.types.is_numeric_dtype(coerced) and set(coerced.dropna().unique()).issubset({0, 1}):
                    df["Churn"] = coerced.astype(int)
                else:
                    # leave as mapped (may contain NaNs) so caller can handle/check
                    df["Churn"] = mapped
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot encoding to categorical variables.

    This function converts categorical columns in the DataFrame into one-hot
    encoded columns and drops the first category level to avoid multicollinearity.

    Args:
        df: Input pandas DataFrame.

    Returns:
        One-hot encoded DataFrame.
    """
    return pd.get_dummies(df, drop_first=True)
