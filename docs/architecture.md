# Project Architecture

## Overview
This project follows a Clean Architecture-style layout with clear separation between data, feature engineering, modeling, evaluation, and application-facing layers. The training entry point remains in the repository root and the reusable business logic lives under the src package.

## Folder Responsibilities
- src/data: data loading and persistence helpers.
- src/features: cleaning, encoding, and scaling transformations.
- src/models: model training, prediction, and artifact serialization logic.
- src/pipelines: orchestration of preprocessing, training, and inference flows.
- src/registry: metadata and artifact registry utilities.
- src/evaluation: metrics, reports, and model comparison helpers.
- artifacts: persisted trained model and metadata assets.
- notebooks: exploratory analysis and experimentation.
- api: production-facing API package skeleton for future deployment.

## Data Flow
1. Raw customer churn data is loaded from data/raw.
2. Preprocessing transforms the dataset and writes pipeline outputs under data/processed.
3. Training uses the prepared features to fit multiple models and evaluate them.
4. The best model and supporting artifacts are persisted under artifacts/.
5. Inference consumes the trained model and aligned feature schema for prediction requests.

## Training Flow
1. The root entry point executes the training pipeline.
2. The preprocessing pipeline loads and prepares the dataset.
3. Multiple models are trained and compared.
4. The best-performing model is registered with metadata and saved to artifacts/.

## Inference Flow
1. Incoming feature data is cleaned and encoded to match training-time columns.
2. The selected model is loaded from artifacts/.
3. The prediction and probability outputs are generated and returned.
