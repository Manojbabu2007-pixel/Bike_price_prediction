"""Lab 6 - champion/challenger lifecycle automation on the MLflow registry.

New versions -> Staging -> best challenger is compared with the Production
champion on R2 (higher is better) -> winner is promoted, losers are archived.
"""
import sys

from mlflow.tracking import MlflowClient

from bike_utils import REGISTERED_MODEL_NAME

METRIC_TO_OPTIMIZE = "r2"


def automate_champion_challenger():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    client = MlflowClient()
    model_name = REGISTERED_MODEL_NAME

    try:
        all_versions = client.search_model_versions(f"name='{model_name}'")

        # 1. New (unstaged) versions -> Staging
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]
        if new_models:
            print(f"\n[INFO] Found {len(new_models)} new model(s). Moving them to Staging...")
            for mv in new_models:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                print(f"  -> Version {mv.version} is now in Staging.")

        all_versions = client.search_model_versions(f"name='{model_name}'")
        staging_models = [mv for mv in all_versions if mv.current_stage == "Staging"]

        if not staging_models:
            print("\n[INFO] No models in Staging to evaluate. Exiting.")
            return

        # 2. Best challenger among Staging
        print("\n[INFO] Evaluating Staging candidates...")
        best_challenger, best_score = None, float("-inf")
        for mv in staging_models:
            score = client.get_run(mv.run_id).data.metrics.get(METRIC_TO_OPTIMIZE, float("-inf"))
            print(f"  -> Staging Candidate: Version {mv.version} | {METRIC_TO_OPTIMIZE}: {score:.4f}")
            if score > best_score:
                best_challenger, best_score = mv, score

        print(f"\n[INFO] Best Challenger is Version {best_challenger.version} ({METRIC_TO_OPTIMIZE}: {best_score:.4f})")

        # 3. Current champion
        production_models = [mv for mv in all_versions if mv.current_stage == "Production"]
        current_champion = production_models[0] if production_models else None
        promote_challenger = False

        if not current_champion:
            print("[INFO] No model currently in Production. Challenger wins by default!")
            promote_challenger = True
        else:
            champ_score = client.get_run(current_champion.run_id).data.metrics.get(METRIC_TO_OPTIMIZE, float("-inf"))
            print(f"[INFO] Current Production Champion: Version {current_champion.version} | {METRIC_TO_OPTIMIZE}: {champ_score:.4f}")

            # 4. Face-off
            if best_score > champ_score:
                print("[SUCCESS] Challenger beat the Champion!")
                promote_challenger = True
            else:
                print("[INFO] Challenger failed to beat the Champion. Champion retains its title.")

        # 5. Promote / archive
        if promote_challenger:
            print(f"\n[INFO] Promoting Version {best_challenger.version} to Production...")
            client.transition_model_version_stage(
                name=model_name, version=best_challenger.version, stage="Production", archive_existing_versions=False
            )
            if current_champion:
                print(f"[INFO] Archiving defeated Champion (Version {current_champion.version})...")
                client.transition_model_version_stage(
                    name=model_name, version=current_champion.version, stage="Archived", archive_existing_versions=False
                )

        # 6. Archive whatever is still in Staging
        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining = [mv for mv in final_versions if mv.current_stage == "Staging"]
        if remaining:
            print("\n[INFO] Cleaning up Staging area...")
            for mv in remaining:
                print(f"  -> Archiving Version {mv.version} (Lost the competition)")
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False
                )

        print("\n[SUCCESS] Automated Model Lifecycle execution complete!")

    except Exception as e:
        print(f"[ERROR] Failed to automate lifecycle: {e}")
        sys.exit(1)


if __name__ == "__main__":
    automate_champion_challenger()
