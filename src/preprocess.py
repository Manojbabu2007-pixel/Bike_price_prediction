"""Lab 4 - preprocessing. Reproduces the notebook (Mani.ipynb) step by step and
saves numpy arrays + the fitted preprocessor for the MLflow tracking scripts."""
import json
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from bike_utils import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, RAW_DATA_PATH, TARGET, TEST_SIZE,
    build_column_transformer, clean_and_engineer, impute_ex_showroom_notebook_style, load_raw,
)


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    if not os.path.exists(RAW_DATA_PATH):
        print(f"[ERROR] Raw data not found at '{RAW_DATA_PATH}'. Copy BIKE DETAILS.csv there.")
        raise SystemExit(1)

    # 1. Load + clean + feature engineering (same order as the notebook)
    df = load_raw()
    rows_raw = len(df)
    df = clean_and_engineer(df)
    df = impute_ex_showroom_notebook_style(df)
    print(f"Rows: {rows_raw} raw -> {len(df)} after duplicate/target cleaning")

    # 2. Split (80/20, random_state=42, as in the notebook)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Impute / encode
    preprocessor = build_column_transformer()
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # 4. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.float64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.float64))
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    metadata = {
        "dataset_name": "Bike Details (used bike selling price)",
        "task": "regression",
        "target": TARGET,
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"X_train: {X_train_final.shape} | X_test: {X_test_final.shape}")
    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
