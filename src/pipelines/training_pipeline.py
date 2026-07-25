from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.models.save_model import save_model
from src.models.train import (
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
)
from src.models.predict import predict
from src.evaluation.metrics import calculate_metrics
from src.evaluation.reports import generate_classification_report
from src.evaluation.comparison import append_model_results, sort_models
from src.pipelines.preprocessing_pipeline import run_preprocessing_pipeline
from src.registry.metadata import create_metadata
from src.registry.model_registry import save_best_model


def run_training_pipeline() -> Tuple[Any, str, pd.DataFrame]:
    """Run the end-to-end training pipeline and persist the best model.

    Workflow:
    1. Execute preprocessing pipeline (loads raw data, preprocesses and
       saves pipeline outputs).
    2. Train Logistic Regression, Decision Tree, and Random Forest models.
    3. Evaluate each model using the reusable evaluation utilities.
    4. Compare models and select the best model by F1 score.
    5. Save the best model to `saved_models/` and print a training summary.

    Returns
    -------
    best_model : Any
        The trained model object selected as best.
    best_name : str
        Human-readable name of the best model.
    comparison_df : pandas.DataFrame
        DataFrame summarizing model metrics for all trained models.
    """
    # 1. Preprocessing
    X_train, X_test, y_train, y_test, scaler = run_preprocessing_pipeline(
        Path("data") / "raw" / "customer_churn.csv"
    )

    # Hold trained models and comparison table
    trained_models: Dict[str, Any] = {}
    comparison_df: pd.DataFrame = pd.DataFrame()

    # 2-4. Train and evaluate models
    trainers = [
        ("Logistic Regression", train_logistic_regression),
        ("Decision Tree", train_decision_tree),
        ("Random Forest", train_random_forest),
    ]

    for name, trainer in trainers:
        model = trainer(X_train, y_train)
        trained_models[name] = model

        # Predictions and metrics
        y_pred = predict(model, X_test)
        metrics = calculate_metrics(y_test, y_pred)

        # Append results for comparison
        comparison_df = append_model_results(comparison_df, name, metrics)

        # Optionally keep textual report in the DataFrame (not printed)
        # report = generate_classification_report(y_test, y_pred)

    # 6. Compare and select best model by F1
    comparison_df = sort_models(comparison_df, metric="f1")
    if comparison_df.empty:
        raise RuntimeError("No model metrics were computed; training may have failed.")

    best_name = comparison_df.iloc[0]["model"]
    best_model = trained_models[best_name]

    # 7. Save best model
    saved_dir = Path("saved_models")
    saved_dir.mkdir(parents=True, exist_ok=True)
    model_filename = saved_dir / f"{best_name.replace(' ', '_')}.joblib"
    save_path = save_model(best_model, model_filename)

    # 8. Save artifacts using the registry helpers
    metrics = comparison_df.loc[comparison_df["model"] == best_name].iloc[0].to_dict()
    metadata = create_metadata(
        model_name=best_name,
        metrics=metrics,
        dataset_name="customer_churn",
    )
    saved_artifacts = save_best_model(
        model=best_model,
        scaler=scaler,
        feature_columns=list(X_train.columns),
        metadata=metadata,
        artifacts_dir=Path("artifacts"),
    )

    # 9. Print training summary
    print("Training summary:")
    print(comparison_df.to_string(index=False))
    print(f"\nSelected best model: {best_name}")
    print(f"Saved best model to: {save_path}")
    print(f"Saved registry artifacts to: {saved_artifacts['model']}")

    return best_model, best_name, comparison_df
