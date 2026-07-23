from pathlib import Path
from typing import Any, Union

import joblib


def load_model(model_path: Union[str, Path]) -> Any:
    """Load a model from disk using joblib.

    Parameters
    ----------
    model_path : str or Path
        Path to the saved model file.

    Returns
    -------
    Any
        The deserialized model object.

    Raises
    ------
    FileNotFoundError
        If the provided `model_path` does not exist or is not a file.
    """
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    model = joblib.load(path)
    return model
