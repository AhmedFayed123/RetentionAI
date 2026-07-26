"""Prediction service for the RetentionAI API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from api.schemas.request import PredictionRequest
from api.schemas.response import PredictionResponse
from src.features.cleaning import convert_total_charges
from src.features.encoding import one_hot_encode
from src.models.load_model import load_model
from src.models.predict import predict, predict_probabilities
from src.registry.model_registry import load_best_model


class PredictionService:
    """Service responsible for loading persisted model artifacts and scoring requests."""

    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._feature_columns: Optional[list[str]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load the persisted model, scaler, feature schema, and metadata."""
        if not self.artifacts_dir.exists():
            raise FileNotFoundError(
                f"Artifacts directory not found: {self.artifacts_dir}; run training pipeline first."
            )

        try:
            model, scaler, feature_columns, metadata = load_best_model(self.artifacts_dir)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not load persisted model artifacts from {self.artifacts_dir}: {exc}"
            ) from exc

        self._model = model
        self._scaler = scaler
        self._feature_columns = list(feature_columns)
        self._metadata = metadata

        if self._model is None:
            raise RuntimeError("Loaded model artifact is invalid or empty.")
        if self._feature_columns is None:
            raise RuntimeError("Loaded feature column schema is invalid or empty.")

    def _prepare_features(self, request: PredictionRequest) -> pd.DataFrame:
        """Convert the request into a feature matrix aligned to the training schema."""
        payload = request.model_dump()
        features = pd.DataFrame([payload])

        # Drop non-feature columns: customerID is an identifier and Churn is
        # the target variable — neither is used as a prediction input.
        non_feature_columns = [col for col in ("customerID", "Churn") if col in features.columns]
        if non_feature_columns:
            features = features.drop(columns=non_feature_columns)

        # Apply the same cleaning/encoding used during training preprocessing.
        features = convert_total_charges(features)
        features = one_hot_encode(features)

        # Align to the persisted feature columns from training. Build the
        # DataFrame in a single pass to avoid pandas fragmentation warnings.
        aligned_data = {}
        for column in self._feature_columns or []:
            aligned_data[column] = features[column] if column in features.columns else 0
        aligned = pd.DataFrame(aligned_data, index=features.index)

        return aligned

    def _apply_scaler(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply the persisted scaler to numeric feature columns when available."""
        if self._scaler is None:
            return features

        # Determine which columns to scale based on the scaler's training-time
        # feature set (feature_names_in_), NOT the current request's data.
        # A single-row request may have columns that look binary (e.g. tenure=1)
        # but were non-binary during training, causing a feature-name mismatch.
        if hasattr(self._scaler, "feature_names_in_"):
            scaling_columns = [
                col for col in self._scaler.feature_names_in_ if col in features.columns
            ]
        else:
            # Fallback: infer from the current request's data
            numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
            binary_columns = [
                column
                for column in numeric_columns
                if set(features[column].dropna().unique()).issubset({0, 1})
                and features[column].nunique() <= 2
            ]
            scaling_columns = [column for column in numeric_columns if column not in binary_columns]

        if scaling_columns:
            scaled_features = features.copy()
            scaled_features[scaling_columns] = self._scaler.transform(scaled_features[scaling_columns])
            return scaled_features

        return features

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Generate a prediction for a single customer request."""
        if self._model is None:
            raise RuntimeError("Prediction model is not available.")

        prepared_features = self._prepare_features(request)
        prepared_features = self._apply_scaler(prepared_features)

        prediction = predict(self._model, prepared_features)
        try:
            probability = predict_probabilities(self._model, prepared_features)
        except Exception as exc:  # pragma: no cover - defensive boundary for unsupported models
            raise AttributeError(
                f"Model does not support probability estimates: {exc}"
            ) from exc

        prediction_value = int(prediction[0])
        probability_value = float(probability[0])
        label = "Churn" if prediction_value == 1 else "No Churn"
        model_name = str(self._metadata.get("model_name", "Unknown")) if self._metadata else "Unknown"

        return PredictionResponse(
            prediction=prediction_value,
            label=label,
            probability=probability_value,
            model_name=model_name,
        )
