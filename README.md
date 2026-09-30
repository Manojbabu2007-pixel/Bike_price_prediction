# Bike Price Prediction – MLOps Labs 4, 5, 6

Regression project (target: `selling_price`) built from `Mani.ipynb`; structure follows the Churn Prediction reference.

## Setup
1. `pip install -r requirements.txt`
2. Copy `BIKE DETAILS.csv` to `data/raw/` (the dataset is not included in this zip).
3. Run every command from the project root.

## Labs
| Lab | Command | What it does |
|---|---|---|
| 4 – MLflow tracking | `python pipelines/run_lab4_tracking.py` | `preprocess.py` (notebook logic) → `train_mlflow.py` (5 tracked runs: R², MAE, RMSE, plots) → `validate_reproducibility.py` |
| 5 – Production pipeline | `python pipelines/run_lab5_pipeline.py` | `validate_data.py` (pandera schema) → `preprocess_pipeline.py` (leakage-safe sklearn Pipeline) → `validate_outputs.py` |
| 6 – Model registry | `python pipelines/run_lab6_registry.py` | `train_registry.py` → `automate_lifecycle.py` (champion/challenger on R²) → `generate_registry_report.py` |

Run Lab 5 before Lab 6. To register a challenger and see a promotion decision:
`python src/train_registry.py --candidate XGBoost_Deep` then `python src/automate_lifecycle.py`.

Outputs: `data/processed/`, `models/`, `artifacts/` (reports/plots), `mlruns/` + `mlflow.db` (view with `mlflow ui`).
