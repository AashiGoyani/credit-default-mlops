# Usage Guide

Complete guide for using the Credit Card Default Prediction MLOps pipeline.

## Table of Contents
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Model Registration](#model-registration)
- [Model Serving](#model-serving)
- [Making Predictions](#making-predictions)
- [Monitoring](#monitoring)
- [MLflow Operations](#mlflow-operations)
- [Advanced Usage](#advanced-usage)

## Data Preparation

### Running Data Preparation
```bash
python src/features/prepare.py
```

### What It Does
1. **Loads raw data** from `src/data/raw/default of credit card clients.xls`
2. **Cleans column names**: Converts to lowercase, replaces spaces with underscores
3. **Engineers features**:
   - Utilization ratios (bill_amt / limit_bal)
   - Payment ratios (pay_amt / bill_amt)
   - Average utilization across 6 months
   - Missed payment count
4. **Splits data**: 80% train, 20% test (stratified)
5. **Saves datasets**:
   - `src/data/processed/train.csv` (24,000 rows)
   - `src/data/processed/test.csv` (6,000 rows)
   - `src/data/reference/reference.csv` (10,000 samples for monitoring)

### Customizing Data Preparation

Edit [src/features/prepare.py](../src/features/prepare.py):

```python
# Change test size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3,  # Change from 0.2 to 0.3
    random_state=42,
    stratify=y
)

# Add custom features
df["custom_feature"] = df["bill_amt1"] * df["age"]
```

### Verifying Prepared Data
```bash
# Check file sizes
ls -lh src/data/processed/

# View first few rows
head -n 5 src/data/processed/train.csv

# Check shape with pandas
python -c "import pandas as pd; df = pd.read_csv('src/data/processed/train.csv'); print(f'Shape: {df.shape}')"
```

## Model Training

### Basic Training
```bash
python src/train/train.py
```

### What It Does
1. **Loads training data** from processed datasets
2. **Creates pipeline**: StandardScaler + LogisticRegression
3. **Trains model** on training set
4. **Evaluates** on test set
5. **Logs to MLflow**:
   - Parameters (model_type)
   - Metrics (accuracy, roc_auc)
   - Artifacts (confusion_matrix.json, model)
6. **Saves model** as sklearn pipeline

### Expected Output
```
✅ MLflow using local tracking
📦 Loading processed datasets...
✅ Pipeline logged | Accuracy=0.8192 | AUC=0.7745

🎯 Training complete! Check MLflow UI for details:
   → http://localhost:5001
```

### Customizing Training

Edit [src/train/train.py](../src/train/train.py):

#### Change Model Parameters
```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(
        max_iter=2000,        # Increase iterations
        solver="lbfgs",       # Change solver
        C=0.5,                # Add regularization
        penalty="l2"
    ))
])
```

#### Try Different Models
```python
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])
```

#### Add Cross-Validation
```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=5, scoring='roc_auc'
)
mlflow.log_metric("cv_auc_mean", cv_scores.mean())
mlflow.log_metric("cv_auc_std", cv_scores.std())
```

### Running Multiple Experiments
```bash
# Run training multiple times
for i in {1..5}; do
    python src/train/train.py
done

# View all runs in MLflow UI
open http://localhost:5001
```

## Model Registration

### Registering Best Model
```bash
python src/train/eval_register.py
```

### What It Does
1. **Searches all runs** in "credit-card-default" experiment
2. **Ranks by ROC-AUC** (descending)
3. **Selects best run**
4. **Registers model** to MLflow Model Registry
5. **Assigns "prod" alias** to the best version

### Expected Output
```
🔍 Fetching experiment runs...
🏆 Best run found: 78f45c930f874efebbca601dd811a61e | ROC_AUC=0.7745
✅ Model 'credit-default-model' registered and aliased as 'prod' (v1)
```

### Managing Model Versions

#### List All Versions
```python
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("file:///path/to/mlruns")
client = MlflowClient()

model_name = "credit-default-model"
versions = client.search_model_versions(f"name='{model_name}'")

for v in versions:
    print(f"Version {v.version}: {v.aliases}")
```

#### Promote a Specific Version
```python
client.set_registered_model_alias(
    model_name,
    "prod",
    version="2"  # Promote version 2
)
```

#### Create Staging Alias
```python
client.set_registered_model_alias(
    model_name,
    "staging",
    version="3"
)
```

## Model Serving

### Starting the API Server

#### Local Development
```bash
uvicorn src.serve.app:app --reload --port 8080
```

#### Production (Docker)
```bash
docker-compose up -d fastapi
```

### API Endpoints

#### 1. Health Check
```bash
curl http://localhost:8080/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-10-28T15:30:00"
}
```

#### 2. API Documentation
Open browser to:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

#### 3. Prometheus Metrics
```bash
curl http://localhost:8080/metrics
```

## Making Predictions

### Single Prediction

#### Using cURL
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_1": 0,
    "PAY_2": -1,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 2,
    "BILL_AMT1": 3913.0,
    "BILL_AMT2": 3102.0,
    "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0,
    "BILL_AMT5": 0.0,
    "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 689.0,
    "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0,
    "PAY_AMT5": 0.0,
    "PAY_AMT6": 0.0
  }'
```

#### Using Python
```python
import requests
import json

url = "http://localhost:8080/predict"

data = {
    "LIMIT_BAL": 20000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_1": 0,
    "PAY_2": -1,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 2,
    "BILL_AMT1": 3913.0,
    "BILL_AMT2": 3102.0,
    "BILL_AMT3": 689.0,
    "BILL_AMT4": 0.0,
    "BILL_AMT5": 0.0,
    "BILL_AMT6": 0.0,
    "PAY_AMT1": 0.0,
    "PAY_AMT2": 689.0,
    "PAY_AMT3": 0.0,
    "PAY_AMT4": 0.0,
    "PAY_AMT5": 0.0,
    "PAY_AMT6": 0.0
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

#### Using Test File
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### Response Format
```json
{
  "prediction": 0,
  "default_probability": 0.1234,
  "timestamp": "2025-10-28T15:30:00.123456",
  "model_uri": "models:/credit-default-model@prod"
}
```

**Fields:**
- `prediction`: Binary prediction (0 = no default, 1 = default)
- `default_probability`: Probability of default (0.0 to 1.0)
- `timestamp`: Prediction time (ISO format)
- `model_uri`: Model used for prediction

### Batch Predictions

#### Python Script
```python
import requests
import pandas as pd
import json

# Load test data
test_df = pd.read_csv('src/data/processed/test.csv')
test_df = test_df.drop(columns=['target', 'id'])

# Make predictions
url = "http://localhost:8080/predict"
predictions = []

for _, row in test_df.head(100).iterrows():
    response = requests.post(url, json=row.to_dict())
    predictions.append(response.json())

# Save results
with open('predictions.json', 'w') as f:
    json.dump(predictions, f, indent=2)

print(f"Made {len(predictions)} predictions")
```

### Understanding Feature Values

#### SEX
- 1 = Male
- 2 = Female

#### EDUCATION
- 1 = Graduate school
- 2 = University
- 3 = High school
- 4 = Others

#### MARRIAGE
- 1 = Married
- 2 = Single
- 3 = Others

#### PAY_* (Payment Status)
- -2 = No credit use
- -1 = Pay duly
- 0 = Paid on time
- 1 = Payment delay for one month
- 2 = Payment delay for two months
- ... (and so on)

## Monitoring

### Prometheus Metrics

#### Query Metrics
```bash
# Total requests
curl http://localhost:8080/metrics | grep request_count_total

# Request latency
curl http://localhost:8080/metrics | grep request_latency_seconds
```

#### Access Prometheus UI
1. Open http://localhost:9090
2. Enter queries:
   - `rate(request_count_total[5m])` - Request rate
   - `histogram_quantile(0.95, request_latency_seconds)` - 95th percentile latency

### Grafana Dashboards

#### Setup Prometheus Data Source
1. Open http://localhost:3000
2. Login: admin/admin
3. Configuration → Data Sources → Add data source
4. Select Prometheus
5. URL: `http://prometheus:9090`
6. Click "Save & Test"

#### Create Dashboard
1. Create → Dashboard
2. Add Panel
3. Query examples:
   ```promql
   # Request Rate
   rate(request_count_total{endpoint="/predict"}[5m])

   # Average Latency
   rate(request_latency_seconds_sum[5m]) / rate(request_latency_seconds_count[5m])

   # Success Rate
   sum(rate(request_count_total{endpoint="/predict"}[5m]))
   ```

#### Import Dashboard
```bash
# Dashboard JSON located at:
# grafana_fastapi_dashboard.json
```

## MLflow Operations

### Accessing MLflow UI
```bash
# Open browser
open http://localhost:5001

# Or with Docker
open http://localhost:5001
```

### Comparing Runs
1. Go to Experiments → credit-card-default
2. Select multiple runs (checkboxes)
3. Click "Compare"
4. View metrics, parameters, and artifacts side-by-side

### Downloading Model
```bash
# Download specific run's model
mlflow artifacts download \
  --run-id <run-id> \
  --artifact-path model \
  --dst-path ./downloaded_model
```

### Loading Model Programmatically
```python
import mlflow.pyfunc

# Load production model
model = mlflow.pyfunc.load_model("models:/credit-default-model@prod")

# Load specific version
model = mlflow.pyfunc.load_model("models:/credit-default-model/1")

# Load from run
model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
```

### Searching Runs
```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search by metric
runs = client.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.roc_auc > 0.75",
    order_by=["metrics.roc_auc DESC"]
)

for run in runs:
    print(f"Run: {run.info.run_id}, AUC: {run.data.metrics['roc_auc']}")
```

## Advanced Usage

### Custom Feature Engineering

Edit [src/serve/app.py](../src/serve/app.py) to add new features:

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Existing features
    df.columns = [c.lower() for c in df.columns]

    # Add custom features
    df["total_bill"] = df[[f"bill_amt{i}" for i in range(1, 7)]].sum(axis=1)
    df["total_payment"] = df[[f"pay_amt{i}" for i in range(1, 7)]].sum(axis=1)
    df["payment_ratio"] = df["total_payment"] / df["total_bill"].replace(0, 1)

    # ... rest of feature engineering
    return df
```

### A/B Testing

#### Register Multiple Models
```python
# Register challenger model
client.set_registered_model_alias("credit-default-model", "challenger", "2")
```

#### Route Traffic
```python
import random

@app.post("/predict")
def predict(data: CreditData):
    # 90% production, 10% challenger
    model_alias = "prod" if random.random() < 0.9 else "challenger"
    model_uri = f"models:/credit-default-model@{model_alias}"

    model = mlflow.pyfunc.load_model(model_uri)
    # ... rest of prediction
```

### Custom Metrics

Add to [src/serve/app.py](../src/serve/app.py):

```python
from prometheus_client import Gauge

PREDICTION_DISTRIBUTION = Gauge(
    'prediction_distribution',
    'Distribution of predictions',
    ['prediction_class']
)

@app.post("/predict")
def predict(data: CreditData):
    # ... existing code

    # Track prediction distribution
    PREDICTION_DISTRIBUTION.labels(prediction_class=str(label)).inc()

    return result
```

### Logging Predictions

```python
import logging
from datetime import datetime

logging.basicConfig(
    filename='predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

@app.post("/predict")
def predict(data: CreditData):
    # ... existing code

    logging.info(f"Prediction: {label}, Probability: {prob}, Input: {data.dict()}")

    return result
```

### Scheduled Retraining

Create `scripts/retrain.sh`:

```bash
#!/bin/bash
# Automated retraining script

echo "Starting retraining..."

# Prepare data
python src/features/prepare.py

# Train model
python src/train/train.py

# Register best model
python src/train/eval_register.py

# Restart API server
docker-compose restart fastapi

echo "Retraining complete!"
```

Set up cron job:
```bash
# Run every Sunday at 2 AM
0 2 * * 0 /path/to/mlops/scripts/retrain.sh
```

## Troubleshooting

### API Returns Error
```bash
# Check API logs
docker-compose logs -f fastapi

# Test with verbose output
curl -v -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### Prediction is Slow
```bash
# Check latency metrics
curl http://localhost:8080/metrics | grep request_latency

# Monitor in Prometheus
# Query: histogram_quantile(0.95, request_latency_seconds)
```

### Model Not Found
```bash
# Check MLflow registry
curl http://localhost:5001/api/2.0/mlflow/registered-models/get?name=credit-default-model

# Verify alias
python -c "from mlflow.tracking import MlflowClient; import mlflow; mlflow.set_tracking_uri('file://mlruns'); client = MlflowClient(); print(client.get_model_version_by_alias('credit-default-model', 'prod'))"
```

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design details
- Explore MLflow UI for experiment tracking
- Set up Grafana dashboards for monitoring
- Implement custom features or models
- Set up CI/CD pipeline for automated deployment
