"""Lab 6 - write a deployment-ready JSON description of the Production model."""
import json
import os
import sys

from mlflow.tracking import MlflowClient

from bike_utils import REGISTERED_MODEL_NAME


def generate_registry_report():
    print("[INFO] Generating Model Registry Report...")

    client = MlflowClient()
    model_name = REGISTERED_MODEL_NAME

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        champion = next((mv for mv in versions if mv.current_stage == "Production"), None)

        if not champion:
            print("[ERROR] No model found in Production stage!")
            sys.exit(1)

        run = client.get_run(champion.run_id)

        report = {
            "registry_status": "READY_FOR_DEPLOYMENT",
            "task": "regression (used bike selling price)",
            "model_lineage": {
                "registered_name": model_name,
                "version": int(champion.version),
                "current_stage": champion.current_stage,
                "run_id": champion.run_id,
                "artifact_uri": champion.source,
            },
            "performance_metrics": {
                "r2": run.data.metrics.get("r2"),
                "mae": run.data.metrics.get("mae"),
                "rmse": run.data.metrics.get("rmse"),
            },
            "hyperparameters": run.data.params,
            "preprocessing_dependency": "models/preprocessor.pkl",
        }

        os.makedirs("artifacts", exist_ok=True)
        report_path = "artifacts/production_model_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"[SUCCESS] Report successfully generated for Version {champion.version}")
        print(f"[INFO] Saved to: {report_path}")

    except SystemExit:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to generate report: {e}")
        sys.exit(1)


if __name__ == "__main__":
    generate_registry_report()
