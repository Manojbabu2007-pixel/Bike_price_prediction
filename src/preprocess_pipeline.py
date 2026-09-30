"""Lab 5 - production preprocessing with a single scikit-learn Pipeline.

Difference from Lab 4: ex-showroom-price medians are learned from the TRAINING
split only (ExShowroomImputer), so nothing from the test set leaks into training.
"""
import json
import os

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from bike_utils import (
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE, RAW_DATA_PATH, TARGET, TEST_SIZE,
    build_production_preprocessor, clean_and_engineer, load_raw,
)


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load + row-wise cleaning / feature engineering (no learned statistics)
    df = clean_and_engineer(load_raw(RAW_DATA_PATH))

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # 2. Split BEFORE any fitted transformation to prevent leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # 3. Fit on train, transform both
    preprocessor = build_production_preprocessor()
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)
    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 4. Save artifacts
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    np.save("data/processed/X_train_final.npy", X_train_final)
    np.save("data/processed/X_test_final.npy", X_test_final)
    np.save("data/processed/y_train.npy", y_train.to_numpy(dtype=np.float64))
    np.save("data/processed/y_test.npy", y_test.to_numpy(dtype=np.float64))

    # NOTE: loading this file needs `src/` on sys.path (ExShowroomImputer lives in bike_utils).
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    metadata = {
        "dataset_name": "Bike Details (used bike selling price)",
        "task": "regression",
        "target": TARGET,
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "imputation": "ex_showroom_price: per-name median (train only), then global median",
    }
    with open("data/processed/dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
