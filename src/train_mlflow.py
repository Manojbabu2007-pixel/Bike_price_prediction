"""Lab 4 - MLflow experiment tracking. One run per model from the notebook."""
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np

from bike_utils import EXPERIMENT_NAME, get_model_configs, regression_metrics

# Models tracked by default (the five compared in the notebook).
DEFAULT_RUNS = ["LinearRegression", "RandomForest", "GradientBoosting", "ExtraTrees", "XGBoost"]


def train_and_track(run_name, family, estimator_cls, params, data):
    X_train, X_test, y_train, y_test = data
    print(f"\n--- Starting MLflow Run: {run_name} ---")

    with mlflow.start_run(run_name=run_name):
        # 1. Parameters
        mlflow.log_params(params)
        mlflow.log_param("model_family", family)

        # 2. Train
        model = estimator_cls(**params)
        model.fit(X_train, y_train)

        # 3. Evaluate + log metrics
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)
        print(f"Metrics logged: R2 = {metrics['r2']:.4f} | MAE = {metrics['mae']:.2f} | RMSE = {metrics['rmse']:.2f}")

        # 4. Diagnostic plots
        os.makedirs("artifacts", exist_ok=True)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_test, y_pred, alpha=0.7)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Actual Selling Price")
        ax.set_ylabel("Predicted Selling Price")
        ax.set_title(f"Actual vs Predicted - {run_name}")
        path = f"artifacts/actual_vs_predicted_{run_name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="plots")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_pred, y_test - y_pred, alpha=0.7)
        ax.axhline(0, color="r", linestyle="--", linewidth=1)
        ax.set_xlabel("Predicted Selling Price")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(f"Residuals - {run_name}")
        path = f"artifacts/residuals_{run_name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(path, artifact_path="plots")

        # 5. Lineage metadata + model
        if os.path.exists("data/processed/dataset_metadata.json"):
            mlflow.log_artifact("data/processed/dataset_metadata.json", artifact_path="metadata")
        mlflow.sklearn.log_model(sk_model=model, name="model")

    return model, metrics


if __name__ == "__main__":
    try:
        data = (
            np.load("data/processed/X_train_final.npy"),
            np.load("data/processed/X_test_final.npy"),
            np.load("data/processed/y_train.npy"),
            np.load("data/processed/y_test.npy"),
        )
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Run src/preprocess.py first.")
        raise SystemExit(1)

    mlflow.set_experiment(EXPERIMENT_NAME)
    configs = get_model_configs()

    best_name, best_model, best_r2 = None, None, float("-inf")
    for name in DEFAULT_RUNS:
        family, cls, params = configs[name]
        model, metrics = train_and_track(name, family, cls, params, data)
        if metrics["r2"] > best_r2:
            best_name, best_model, best_r2 = name, model, metrics["r2"]

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\nBest run: {best_name} (R2 = {best_r2:.4f}) saved to models/best_model.pkl")
