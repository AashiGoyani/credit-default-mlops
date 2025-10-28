# __define-ocg__: Evaluate runs and register best Credit Card Default model
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

EXPERIMENT_NAME = "credit-card-default"
MODEL_NAME = "credit-default-model"

mlflow.set_tracking_uri(f"file://{Path('mlruns').absolute()}")
mlflow.set_registry_uri(f"file://{Path('mlruns').absolute()}")

client = MlflowClient()

print("🔍 Fetching experiment runs...")
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    raise ValueError(f"Experiment '{EXPERIMENT_NAME}' not found. Run train.py first.")

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.roc_auc DESC"],
    max_results=5,
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_auc = best_run.data.metrics.get("roc_auc")
print(f"🏆 Best run found: {best_run_id} | ROC_AUC={best_auc:.4f}")

model_uri = f"runs:/{best_run_id}/model"
model_version = mlflow.register_model(model_uri, MODEL_NAME)

client.set_registered_model_alias(MODEL_NAME, "prod", model_version.version)
print(f"✅ Model '{MODEL_NAME}' registered and aliased as 'prod' (v{model_version.version})")
