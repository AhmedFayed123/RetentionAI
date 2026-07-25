from __future__ import annotations

from datetime import datetime
from platform import python_version
from typing import Any, Dict

import sklearn


def create_metadata(
    model_name: str,
    metrics: Dict[str, float],
    dataset_name: str,
) -> Dict[str, Any]:
    """Create a standardized metadata dictionary for a trained model."""
    return {
        "model_name": model_name,
        "accuracy": metrics.get("accuracy"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "dataset": dataset_name,
        "training_date": datetime.utcnow().isoformat() + "Z",
        "python_version": python_version(),
        "sklearn_version": sklearn.__version__,
    }
