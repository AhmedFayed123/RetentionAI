import json
from pathlib import Path
nb = {
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Project Overview\n",
        "\n",
        "This notebook preprocesses the RetentionAI dataset for binary classification. It loads the data, validates quality, encodes features, scales numeric inputs, and performs a train/test split.\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Import Libraries\n",
        "\n",
        "Group all imports for the preprocessing workflow.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "from pathlib import Path\n",
        "import sys\n",
        "\n",
        "import joblib\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "project_root = Path.cwd()\n",
        "if not (project_root / 'src').is_dir():\n",
        "    project_root = project_root.parent\n",
        "\n",
        "if str(project_root) not in sys.path:\n",
        "    sys.path.insert(0, str(project_root))\n",
        "\n",
        "from src.data.load_data import load_dataset\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Load Dataset\n",
        "\n",
        "Load the dataset and display the first rows and the dataset shape.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "df = load_dataset()\n",
        "\n",
        "df.head()\n",
        "print(df.shape)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Dataset Validation\n",
        "\n",
        "Validate the loaded dataset structure before preprocessing.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "df.info()\n",
        "print(f\"Current shape: {df.shape}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Convert TotalCharges to Numeric\n",
        "\n",
        "Convert `TotalCharges` to numeric values and inspect the data type before and after conversion.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(f\"Current data type: {df['TotalCharges'].dtype}\")\n",
        "\n",
        "df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')\n",
        "\n",
        "print(f\"New data type: {df['TotalCharges'].dtype}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Missing Values Analysis\n",
        "\n",
        "Review missing values by count and percentage.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "missing_counts = df.isna().sum()\n",
        "missing_summary = pd.DataFrame({\n",
        "    'missing_count': missing_counts,\n",
        "    'missing_percentage': (missing_counts / len(df) * 100).round(2),\n",
        "}).sort_values('missing_count', ascending=False)\n",
        "\n",
        "display(missing_summary)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Handle Missing Values\n",
        "\n",
        "Drop rows with missing values and reset the index.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "rows_before = len(df)\n",
        "df = df.dropna().reset_index(drop=True)\n",
        "rows_removed = rows_before - len(df)\n",
        "\n",
        "print(f\"Rows removed: {rows_removed}\")\n",
        "print(f\"Updated dataset shape: {df.shape}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Duplicate Validation\n",
        "\n",
        "Check for duplicate rows after handling missing values.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "duplicate_rows = df.duplicated().sum()\n",
        "print(f\"Duplicate rows: {duplicate_rows}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Categorical Feature Detection\n",
        "\n",
        "Detect categorical columns while excluding identifiers and the target.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "target_column = 'Churn'\n",
        "excluded_columns = ['customerID', target_column]\n",
        "\n",
        "categorical_features = (\n",
        "    df.select_dtypes(include=['object', 'category', 'bool']).columns\n",
        "    .drop(excluded_columns, errors='ignore')\n",
        "    .tolist()\n",
        ")\n",
        "\n",
        "print('Categorical features:', categorical_features)\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# One-Hot Encoding\n",
        "\n",
        "One-hot encode the identified categorical features and report the dataset shape before and after encoding.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "shape_before_encoding = df.shape\n",
        "df = pd.get_dummies(df, columns=categorical_features, drop_first=True)\n",
        "shape_after_encoding = df.shape\n",
        "\n",
        "print(f\"Dataset shape before encoding: {shape_before_encoding}\")\n",
        "print(f\"Dataset shape after encoding: {shape_after_encoding}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Prepare Target Variable\n",
        "\n",
        "Convert the target column to binary numeric labels.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "target_column = 'Churn'\n",
        "\n",
        "binary_mapping = {'No': 0, 'Yes': 1}\n",
        "df[target_column] = df[target_column].map(binary_mapping)\n",
        "\n",
        "print('Unique target values after conversion:', df[target_column].unique())\n",
        "print(df[target_column].value_counts().sort_index())\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Split Features and Target\n",
        "\n",
        "Separate features and the prediction target.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "X = df.drop(columns=[target_column])\n",
        "y = df[target_column].copy()\n",
        "\n",
        "print(f\"X shape: {X.shape}\")\n",
        "print(f\"y shape: {y.shape}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Train/Test Split\n",
        "\n",
        "Split the dataset while preserving the target distribution.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X,\n",
        "    y,\n",
        "    test_size=0.2,\n",
        "    random_state=42,\n",
        "    stratify=y,\n",
        ")\n",
        "\n",
        "print(f\"X_train shape: {X_train.shape}\")\n",
        "print(f\"X_test shape: {X_test.shape}\")\n",
        "print(f\"y_train shape: {y_train.shape}\")\n",
        "print(f\"y_test shape: {y_test.shape}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Feature Scaling\n",
        "\n",
        "Scale numeric features using a scaler fitted only on the training set.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "numeric_columns = X_train.select_dtypes(include=[np.number]).columns.tolist()\n",
        "binary_columns = [\n",
        "    col for col in numeric_columns\n",
        "    if X_train[col].dropna().isin([0, 1]).all() and X_train[col].nunique() <= 2\n",
        "]\n",
        "scaling_columns = [col for col in numeric_columns if col not in binary_columns]\n",
        "\n",
        "scaler = StandardScaler()\n",
        "X_train_scaled = X_train.copy()\n",
        "X_test_scaled = X_test.copy()\n",
        "\n",
        "X_train_scaled[scaling_columns] = scaler.fit_transform(X_train[scaling_columns])\n",
        "X_test_scaled[scaling_columns] = scaler.transform(X_test[scaling_columns])\n",
        "\n",
        "artifacts_dir = Path('artifacts')\n",
        "scaler_path = artifacts_dir / 'scaler.pkl'\n",
        "artifacts_dir.mkdir(parents=True, exist_ok=True)\n",
        "joblib.dump(scaler, scaler_path)\n",
        "\n",
        "print(f\"X_train_scaled shape: {X_train_scaled.shape}\")\n",
        "print(f\"X_test_scaled shape: {X_test_scaled.shape}\")\n",
        "print(f\"Saved StandardScaler to {scaler_path.resolve()}\")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Final Validation\n",
        "\n",
        "Verify the final preprocessing outputs before moving to modeling.\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "validation_results = []\n",
        "\n",
        "validation_results.append((\'No missing values in X_train_scaled\', X_train_scaled.isna().sum().sum() == 0))\n",
        "validation_results.append((\'No missing values in X_test_scaled\', X_test_scaled.isna().sum().sum() == 0))\n",
        "validation_results.append((\'No missing values in y_train\', y_train.isna().sum() == 0))\n",
        "validation_results.append((\'No missing values in y_test\', y_test.isna().sum() == 0))\n",
        "validation_results.append((\'No object columns remain in X_train_scaled\', X_train_scaled.select_dtypes(include=[\'object\']).shape[1] == 0))\n",
        "validation_results.append((\'X_train_scaled and X_test_scaled have identical columns\', X_train_scaled.columns.equals(X_test_scaled.columns)))\n",
        "validation_results.append((\'Target is binary\', set(y_train.unique()) <= {0, 1} and set(y_test.unique()) <= {0, 1}))\n",
        "\n",
        "for message, passed in validation_results:\n",
        "    print(f\"[{\'PASS\' if passed else \'FAIL\'}] {message}\")\n",
        "\n",
        "print(\"\nFinal dataset shapes:\")\n",
        "print(f\"X_train_scaled: {X_train_scaled.shape}\")\n",
        "print(f\"X_test_scaled: {X_test_scaled.shape}\")\n",
        "print(f\"y_train: {y_train.shape}\")\n",
        "print(f\"y_test: {y_test.shape}\")\n"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
Path(r'd:\cv projects\RetentionAI\notebooks\02_preprocessing.ipynb').write_text(json.dumps(nb, indent=1), encoding='utf-8')
