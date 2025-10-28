# __define-ocg__: Train and log Credit Card Default model with MLflow
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

# =====================================================
# ⚙️ MLflow Setup
# =====================================================
mlflow.set_tracking_uri(f"file://{Path('mlruns').absolute()}")
experiment = mlflow.set_experiment("credit-card-default")
print("✅ MLflow using local tracking")

# =====================================================
# 📂 Load Datasets
# =====================================================
PROC_PATH = Path("src/data/processed")
TRAIN_PATH = PROC_PATH / "train.csv"
TEST_PATH = PROC_PATH / "test.csv"

print("📦 Loading processed datasets...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# =====================================================
# 🚀 Train Pipeline + Log
# =====================================================
with mlflow.start_run(run_name="logreg_pipeline"):
    # Combine scaler + model into one pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, solver="liblinear"))
    ])

    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # Log metrics
    mlflow.log_param("model_type", "Pipeline(LogisticRegression+Scaler)")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)
    mlflow.log_dict({"confusion_matrix": cm.tolist()}, "confusion_matrix.json")

    # Log full pipeline as a single model
    mlflow.sklearn.log_model(pipeline, artifact_path="model")

    print(f"✅ Pipeline logged | Accuracy={acc:.4f} | AUC={auc:.4f}")

print("\n🎯 Training complete! Check MLflow UI for details:")
print("   → http://localhost:5001")
