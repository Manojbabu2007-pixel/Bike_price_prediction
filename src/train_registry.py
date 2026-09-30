"""Lab 6 - train a candidate model and register it in the MLflow Model Registry.

Run with no arguments to register the notebook's XGBoost model (version 1).
Register a challenger afterwards with:  python src/train_registry.py --candidate XGBoost_Deep
"""
import argparse

import mlflow
import mlflow.sklearn
import numpy as np

from bike_utils import EXPERIMENT_NAME, REGISTERED_MODEL_NAME, get_model_configs, regression_metrics


def train_and_register_model(candidate="XGBoost"):
    print("[INFO] --- Starting MLflow Run with Model Registry ---")
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        X_train = np.load("data/processed/X_train_final.npy")
        X_test = np.load("data/processed/X_test_final.npy")
        y_train = np.load("data/processed/y_train.npy")
        y_test = np.load("data/processed/y_test.npy")
    except FileNotFoundError:
        print("[ERROR] Processed data not found. Please run the Lab 5 pipeline first.")
        raise SystemExit(1)

    configs = get_model_configs()
    if candidate not in configs:
        print(f"[ERROR] Unknown candidate '{candidate}'. Choose from: {', '.join(configs)}")
        raise SystemExit(1)
    family, estimator_cls, params = configs[candidate]

    with mlflow.start_run(run_name=f"{candidate}_Registry"):
        mlflow.log_params(params)
        mlflow.log_param("model_family", family)
        mlflow.log_param("candidate", candidate)

        print(f"[INFO] Training {candidate} model...")
        model = estimator_cls(**params)
        model.fit(X_train, y_train)

        print("[INFO] Evaluating model...")
        metrics = regression_metrics(y_test, model.predict(X_test))
        mlflow.log_metrics(metrics)

        print("[INFO] Attaching preprocessing pipeline to model artifacts...")
        mlflow.log_artifact("models/preprocessor.pkl", artifact_path="preprocessing_pipeline")

        print("[INFO] Pushing model to MLflow Registry...")
        mlflow.sklearn.log_model(
            sk_model=model,
            name="bike_price_model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"[SUCCESS] Metrics logged: R2 = {metrics['r2']:.4f} | MAE = {metrics['mae']:.2f} | RMSE = {metrics['rmse']:.2f}")
        print(f"[SUCCESS] Model successfully registered under name: '{REGISTERED_MODEL_NAME}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="XGBoost", help="Model config name from bike_utils.get_model_configs()")
    train_and_register_model(parser.parse_args().candidate)
