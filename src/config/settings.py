"""Central filesystem paths for the RetentionAI project."""

from pathlib import Path


# Absolute path to the repository root, derived from this module's location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directory containing source data files.
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
# Directory containing intermediate data created during preparation.
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"
# Directory containing cleaned, model-ready datasets.
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Default location of the project's raw dataset.
RAW_DATA_PATH = RAW_DATA_DIR / "customer_churn.csv"

# Directory containing generated project artifacts.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
# Directory containing saved machine learning models.
MODELS_DIR = ARTIFACTS_DIR / "models"
# Directory containing generated reports.
REPORTS_DIR = ARTIFACTS_DIR / "reports"
# Directory containing generated figures and visualizations.
FIGURES_DIR = ARTIFACTS_DIR / "figures"

# Directory containing application and experiment logs.
LOGS_DIR = PROJECT_ROOT / "logs"
# Directory containing exploratory and analytical notebooks.
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
