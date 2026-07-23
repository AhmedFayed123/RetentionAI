from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from src.data.load_data import load_dataset
from src.data.save_data import save_dataframe
from src.features.cleaning import (
    convert_total_charges,
    remove_duplicates,
    remove_missing_values,
    reset_dataframe_index,
)
from src.features.encoding import encode_target, one_hot_encode
from src.features.scaling import scale_features
from src.features.split import split_dataset


def run_preprocessing_pipeline(
    input_path: Union[str, Path]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, object]:
    """Run the end-to-end preprocessing pipeline and persist outputs.

    The pipeline performs the following steps in order:
    1. Load dataset from `input_path`.
    2. Clean dataset (convert types, drop missing, remove duplicates).
    3. Encode the target column (`Churn`).
    4. One-hot encode categorical variables.
    5. Split into train/test partitions (stratified).
    6. Scale numeric features (fit scaler on training set).
    7. Save the fully preprocessed dataset and the pipeline outputs
       (`X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`).

    Parameters
    ----------
    input_path : str | Path
        Path to the raw CSV input file.

    Returns
    -------
    X_train : pandas.DataFrame
        Scaled training features.
    X_test : pandas.DataFrame
        Scaled test features.
    y_train : pandas.Series
        Training targets.
    y_test : pandas.Series
        Test targets.
    scaler : object
        Fitted scaler instance returned by `scale_features` (e.g. StandardScaler).
    """
    raw_path = Path(input_path)
    df = load_dataset(str(raw_path))

    # Cleaning
    df = convert_total_charges(df)
    df = remove_missing_values(df)
    df = remove_duplicates(df)
    df = reset_dataframe_index(df)

    # Encoding
    df = encode_target(df)
    df = one_hot_encode(df)

    # Split features / target
    target_column = "Churn"
    if target_column not in df.columns:
        raise KeyError(f"Expected target column '{target_column}' not found after encoding.")

    X = df.drop(columns=[target_column])
    y = df[target_column].copy()

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Scale numeric features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Persist datasets
    processed_dir = Path.cwd() / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    processed_file = processed_dir / "preprocessed_retention.csv"
    preprocessed_df = pd.concat([X, y], axis=1)
    save_dataframe(preprocessed_df, str(processed_file))

    pipeline_output_dir = processed_dir / "pipeline_outputs"
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    save_dataframe(X_train_scaled, str(pipeline_output_dir / "X_train.csv"))
    save_dataframe(X_test_scaled, str(pipeline_output_dir / "X_test.csv"))
    save_dataframe(y_train.to_frame(name=target_column), str(pipeline_output_dir / "y_train.csv"))
    save_dataframe(y_test.to_frame(name=target_column), str(pipeline_output_dir / "y_test.csv"))

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
