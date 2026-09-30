"""Lab 4 - train the same model twice with fixed seeds and prove the results match."""
import json
import os

import numpy as np

from bike_utils import get_model_configs, regression_metrics


def run_deterministic_test():
    print("Running Reproducibility Validation...")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    family, cls, params = get_model_configs()["XGBoost"]

    model_1 = cls(**params).fit(X_train, y_train)
    pred_1 = model_1.predict(X_test)
    r2_1 = regression_metrics(y_test, pred_1)["r2"]

    model_2 = cls(**params).fit(X_train, y_train)
    pred_2 = model_2.predict(X_test)
    r2_2 = regression_metrics(y_test, pred_2)["r2"]

    max_diff = float(np.max(np.abs(pred_1 - pred_2)))
    is_reproducible = (r2_1 == r2_2) and max_diff == 0.0

    report = {
        "test_name": "Pipeline Reproducibility Validation",
        "model": family,
        "parameters": params,
        "execution_1_r2": r2_1,
        "execution_2_r2": r2_2,
        "max_abs_prediction_difference": max_diff,
        "is_strictly_reproducible": bool(is_reproducible),
        "status": "PASSED" if is_reproducible else "FAILED",
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/reproducibility_report.json", "w") as f:
        json.dump(report, f, indent=4)

    print(f"Execution 1 R2: {r2_1:.6f}")
    print(f"Execution 2 R2: {r2_2:.6f}")
    if is_reproducible:
        print("SUCCESS: Pipeline is 100% reproducible. Report saved.")
    else:
        print("FAILED: Pipeline is non-deterministic.")
        raise SystemExit(1)


if __name__ == "__main__":
    run_deterministic_test()
