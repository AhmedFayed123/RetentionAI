from pathlib import Path
from typing import Any, Tuple, Union

import joblib
import numpy as np
import pandas as pd

from src.features.cleaning import convert_total_charges
from src.features.encoding import one_hot_encode
from src.models.load_model import load_model
from src.models.predict import predict, predict_probabilities


def _find_latest_model(saved_dir: Path) -> Path:
    models = list(saved_dir.glob("*.joblib"))
    if not models:
        raise FileNotFoundError(f"No saved model files found in {saved_dir}")
    # pick most recently modified
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0]


def _find_scaler_possibilities() -> list:
    # common locations used by notebooks and workflows
    return [
        Path("artifacts") / "best_scaler.joblib",
        Path("artifacts") / "scaler.pkl",
        Path("notebooks") / "artifacts" / "scaler.pkl",
        Path("data") / "processed" / "scaler.pkl",
    ]


def predict_customer(data: Union[dict, pd.Series, pd.DataFrame]) -> Tuple[int, float]:
    """Predict churn for a single customer input.

    Parameters
    ----------
    data : dict | pandas.Series | pandas.DataFrame
        Single-customer feature values. If a DataFrame is provided, only the
        first row is used.

    Returns
    -------
    prediction : int
        Predicted class label (0 or 1).
    probability : float
        Probability of the positive class (1).

    Raises
    ------
    FileNotFoundError
        If no saved model is found.
    AttributeError
        If the model does not support probability prediction.
    """
    # Normalize input to single-row DataFrame
    if isinstance(data, dict):
        X_in = pd.DataFrame([data])
    elif isinstance(data, pd.Series):
        X_in = data.to_frame().T
    elif isinstance(data, pd.DataFrame):
        if data.shape[0] < 1:
            raise ValueError("Input DataFrame must have at least one row")
        X_in = data.iloc[[0]].copy()
    else:
        raise TypeError("`data` must be a dict, pandas.Series, or pandas.DataFrame")

    # Basic cleaning and encoding
    X_in = convert_total_charges(X_in)
    X_in = one_hot_encode(X_in)

    # Load training columns to align features if available
    train_features_path = Path("data") / "processed" / "pipeline_outputs" / "X_train.csv"
    training_columns = None
    if train_features_path.exists():
        train_sample = pd.read_csv(train_features_path)
        training_columns = list(train_sample.columns)

    if training_columns is not None:
        # Reindex to training columns, filling missing with 0
        X_aligned = X_in.reindex(columns=training_columns, fill_value=0)
    else:
        X_aligned = X_in

    # Load the persisted best model from the production artifacts directory
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        raise FileNotFoundError("`artifacts` directory not found; run training pipeline first.")

    model_path = artifacts_dir / "best_model.joblib"
    if not model_path.exists():
        model_path = _find_latest_model(artifacts_dir)

    model = load_model(model_path)

    # Try to load a scaler from common locations; scaler is optional but
    # required for numeric scaling if present
    scaler = None
    for p in _find_scaler_possibilities():
        if p.exists():
            try:
                scaler = joblib.load(p)
            except Exception:
                scaler = None
            if scaler is not None:
                break

    # If scaler found and training columns present, apply scaling to numeric cols
    if scaler is not None and training_columns is not None:
        # Determine numeric columns from training sample
        numeric_cols = train_sample.select_dtypes(include=[np.number]).columns.tolist()
        # Identify binary indicator columns to exclude from scaling
        binary_cols = [
            c
            for c in numeric_cols
            if set(train_sample[c].dropna().unique()).issubset({0, 1}) and train_sample[c].nunique() <= 2
        ]
        scaling_cols = [c for c in numeric_cols if c not in binary_cols and c in X_aligned.columns]
        if scaling_cols:
            X_aligned[scaling_cols] = scaler.transform(X_aligned[scaling_cols])

    # Predict
    y_pred = predict(model, X_aligned)
    try:
        y_prob = predict_probabilities(model, X_aligned)
    except Exception as e:
        raise AttributeError(
            f"Model does not support probability estimates: {e}"
        )

    # Return scalar values
    return int(y_pred[0]), float(y_prob[0])
