# __define-ocg__: FastAPI app serving the Credit Default Production model
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import pandas as pd
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from pathlib import Path
import time

# -------------------------
# 📊 Prometheus metrics
# -------------------------
REQUEST_COUNT = Counter('request_count_total', 'Total number of requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency (seconds)', ['endpoint'])

# -------------------------
# ⚙️ Load Model (Pipeline)
# -------------------------
mlruns_path = Path("mlruns").resolve().as_uri()
mlflow.set_tracking_uri(mlruns_path)
mlflow.set_registry_uri(mlruns_path)

MODEL_URI = "models:/credit-default-model@prod"
print(f"🔄 Loading Production model from MLflow: {MODEL_URI}")
model = mlflow.pyfunc.load_model(MODEL_URI)
print("✅ Pipeline model (scaler + logistic regression) loaded successfully!")

# -------------------------
# 🚀 FastAPI App
# -------------------------
app = FastAPI(
    title="Credit Default Prediction API",
    description="Predicts default probability for credit card customers using the Production MLflow model pipeline.",
    version="1.0.0",
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    REQUEST_COUNT.labels(request.method, request.url.path).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(time.time() - start)
    return response

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")

class CreditData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_1: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.get("/health")
def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as in prepare.py"""
    # Convert column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Add id column if not present (model expects it)
    if 'id' not in df.columns:
        df.insert(0, 'id', 0)

    # Utilization ratios and payment ratios
    for i in range(1, 7):
        df[f"util_{i}"] = df[f"bill_amt{i}"] / df["limit_bal"].replace(0, 1)
        df[f"pay_ratio_{i}"] = df[f"pay_amt{i}"] / df[f"bill_amt{i}"].replace(0, 1)

    # Average utilization across 6 months
    df["avg_util"] = df[[f"util_{i}" for i in range(1, 7)]].mean(axis=1)

    # Missed payment count
    df["misspay_cnt"] = (df[[f"pay_{i}" for i in range(1, 7)]] > 0).sum(axis=1)

    return df

@app.post("/predict")
def predict(data: CreditData):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data.model_dump()])

        # Apply feature engineering
        df = engineer_features(df)

        # MLflow sklearn models support predict_proba through the wrapper
        # Try multiple ways to access predict_proba
        if hasattr(model, 'predict_proba'):
            # Direct method on wrapper
            proba = model.predict_proba(df)
        elif hasattr(model, '_model_impl') and hasattr(model._model_impl, 'predict_proba'):
            # Through _model_impl
            proba = model._model_impl.predict_proba(df)
        else:
            raise AttributeError("Model does not support predict_proba")

        prob = float(proba[0, 1])
        label = int(prob >= 0.5)

        return {
            "prediction": label,
            "default_probability": round(prob, 4),
            "timestamp": datetime.now().isoformat(),
            "model_uri": MODEL_URI
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "message": "Prediction failed",
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat()
        }
