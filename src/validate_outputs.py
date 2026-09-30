"""Lab 5 - sanity checks on the preprocessing outputs + JSON summary report."""
import json
import os

import numpy as np


def validate_preprocessing_outputs():
    print("[INFO] Validating Preprocessing Outputs...")

    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError as e:
        print(f"[ERROR] Could not find processed data: {e}")
        return False

    errors = []

    # Missing / infinite values (validates the imputers and the log features)
    for name, arr in [("X_train", X_train), ("X_test", X_test)]:
        if np.isnan(arr).sum() > 0:
            errors.append(f"NaNs detected in {name} after preprocessing.")
        if np.isinf(arr).sum() > 0:
            errors.append(f"Infinite values detected in {name} after preprocessing.")

    # Shape consistency (validates one-hot encoding and the split)
    if X_train.shape[1] != X_test.shape[1]:
        errors.append(f"Feature mismatch: X_train has {X_train.shape[1]} cols, X_test has {X_test.shape[1]} cols.")
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Row mismatch between X_train and y_train.")
    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Row mismatch between X_test and y_test.")

    # Regression target checks
    for name, arr in [("y_train", y_train), ("y_test", y_test)]:
        if not np.isfinite(arr).all():
            errors.append(f"Non-finite values in {name}.")
        elif (arr <= 0).any():
            errors.append(f"Non-positive selling prices in {name}.")

    if not os.path.exists("models/preprocessor.pkl"):
        errors.append("models/preprocessor.pkl was not saved.")

    report = {
        "validation_status": "PASSED" if not errors else "FAILED",
        "matrix_dimensions": {
            "X_train_shape": list(X_train.shape),
            "X_test_shape": list(X_test.shape),
            "y_train_shape": list(y_train.shape),
            "y_test_shape": list(y_test.shape),
        },
        "data_quality": {
            "missing_values_X_train": int(np.isnan(X_train).sum()),
            "missing_values_X_test": int(np.isnan(X_test).sum()),
        },
        "target_summary": {
            "train_min": float(y_train.min()), "train_max": float(y_train.max()),
            "train_mean": float(y_train.mean()),
            "test_min": float(y_test.min()), "test_max": float(y_test.max()),
            "test_mean": float(y_test.mean()),
        },
        "errors": errors,
    }

    os.makedirs("artifacts", exist_ok=True)
    report_path = "artifacts/preprocessing_summary_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("[ERROR] Output Validation FAILED!")
        for e in errors:
            print(f"  - {e}")
        return False

    print("[SUCCESS] Output Validation PASSED.")
    print("[INFO] Features are clean, encoded, and dimensionally consistent.")
    print(f"[INFO] Summary report saved to {report_path}")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if validate_preprocessing_outputs() else 1)
