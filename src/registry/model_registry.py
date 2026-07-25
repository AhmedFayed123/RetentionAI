from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib


def _resolve_artifacts_dir(artifacts_dir: str | Path) -> Path:
    """Resolve the artifacts directory and create it if missing."""
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    return artifacts_path


def _save_joblib_artifact(payload: Any, path: Path) -> Path:
    """Persist a Python object to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, path)
    return path


def _load_joblib_artifact(path: Path, description: str) -> Any:
    """Load a joblib artifact and raise a descriptive error if unavailable."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing artifact: {description} at {path}")

    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise RuntimeError(f"Failed to load {description} from {path}: {exc}") from exc


def save_best_model(
    model: Any,
    scaler: Any,
    feature_columns: Any,
    metadata: Dict[str, Any],
    artifacts_dir: str | Path = "artifacts",
) -> Dict[str, Path]:
    """Persist the best model artifacts to disk.

    The function saves the trained model, scaler, feature column list, and
    metadata under the specified artifacts directory.
    """
    artifacts_path = _resolve_artifacts_dir(artifacts_dir)

    model_path = _save_joblib_artifact(model, artifacts_path / "best_model.joblib")
    scaler_path = _save_joblib_artifact(scaler, artifacts_path / "best_scaler.joblib")
    feature_columns_path = _save_joblib_artifact(
        feature_columns,
        artifacts_path / "feature_columns.joblib",
    )

    metadata_path = artifacts_path / "metadata.json"
    try:
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    except TypeError as exc:
        raise ValueError(f"Metadata must be JSON-serializable: {exc}") from exc

    return {
        "model": model_path,
        "scaler": scaler_path,
        "feature_columns": feature_columns_path,
        "metadata": metadata_path,
    }


def load_best_model(
    artifacts_dir: str | Path = "artifacts",
) -> Tuple[Any, Any, Any, Any]:
    """Load the persisted best-model artifacts from disk."""
    artifacts_path = Path(artifacts_dir)
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_path}")

    model_path = artifacts_path / "best_model.joblib"
    scaler_path = artifacts_path / "best_scaler.joblib"
    feature_columns_path = artifacts_path / "feature_columns.joblib"
    metadata_path = artifacts_path / "metadata.json"

    model = _load_joblib_artifact(model_path, "model")
    scaler = _load_joblib_artifact(scaler_path, "scaler")
    feature_columns = _load_joblib_artifact(feature_columns_path, "feature columns")

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing artifact: metadata at {metadata_path}")

    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Metadata file is not valid JSON: {metadata_path}") from exc

    return model, scaler, feature_columns, metadata
