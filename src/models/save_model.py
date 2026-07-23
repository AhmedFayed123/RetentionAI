from pathlib import Path
from typing import Any, Union

import joblib


def save_model(model: Any, output_path: Union[str, Path]) -> Path:
    """Save a fitted model to disk using joblib.

    Parameters
    ----------
    model : Any
        The fitted model object to persist.
    output_path : str or Path
        File path where the model will be saved. Parent directories will be
        created if they do not exist.

    Returns
    -------
    pathlib.Path
        The absolute path to the saved model file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    saved_path = path.resolve()
    print(f"Saved model to: {saved_path}")
    return saved_path
