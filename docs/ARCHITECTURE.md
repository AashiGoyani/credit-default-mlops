# Architecture Documentation

Technical architecture and design decisions for the Credit Card Default Prediction MLOps pipeline.

## Table of Contents
- [System Overview](#system-overview)
- [Architecture Diagram](#architecture-diagram)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Model Pipeline](#model-pipeline)
- [API Design](#api-design)
- [Monitoring Stack](#monitoring-stack)
- [Infrastructure](#infrastructure)
- [Security Considerations](#security-considerations)
- [Scalability](#scalability)

## System Overview

The system is a complete MLOps pipeline implementing the full machine learning lifecycle from data ingestion to production serving with monitoring.

### Design Principles
- **Reproducibility**: All experiments tracked with MLflow
- **Automation**: Docker-based deployment
- **Observability**: Prometheus metrics and Grafana dashboards
- **Modularity**: Separate concerns (data, training, serving)
- **Version Control**: Model versioning and aliasing

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| ML Framework | scikit-learn | Model training |
| Experiment Tracking | MLflow | Logging, versioning, registry |
| API Framework | FastAPI | RESTful API serving |
| Web Server | Uvicorn | ASGI server |
| Metrics | Prometheus | Time-series metrics |
| Visualization | Grafana | Dashboards |
| Database | PostgreSQL | MLflow backend store |
| Containerization | Docker | Service isolation |
| Orchestration | Docker Compose | Multi-container management |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Development Phase                        │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Raw Data   │─────▶│   prepare.py │─────▶│  Processed   │
│  (Excel)     │      │  (Features)  │      │  Data (CSV)  │
└──────────────┘      └──────────────┘      └──────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │  train.py    │
                                            │  (Training)  │
                                            └──────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │   MLflow     │
                                            │  Tracking    │
                                            └──────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────┐
                                            │eval_register │
                                            │   .py        │
                                            └──────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Production Phase                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        Docker Compose Network                     │
│                                                                   │
│  ┌──────────────┐          ┌──────────────┐                     │
│  │  PostgreSQL  │◀─────────│   MLflow     │                     │
│  │   Database   │          │   Server     │                     │
│  │  :5432       │          │   :5001      │                     │
│  └──────────────┘          └──────────────┘                     │
│         │                          │                             │
│         │                          ▼                             │
│         │                  ┌──────────────┐                     │
│         │                  │   FastAPI    │◀────────┐           │
│         │                  │   Server     │         │           │
│         │                  │   :8080      │         │           │
│         │                  └──────────────┘         │           │
│         │                          │                 │           │
│         │                          │ /metrics        │           │
│         │                          ▼                 │           │
│         │                  ┌──────────────┐         │           │
│         │                  │  Prometheus  │─────────┘           │
│         │                  │   :9090      │                     │
│         │                  └──────────────┘                     │
│         │                          │                             │
│         │                          ▼                             │
│         │                  ┌──────────────┐                     │
│         │                  │   Grafana    │                     │
│         │                  │   :3000      │                     │
│         │                  └──────────────┘                     │
│         │                                                        │
│         │                  ┌──────────────┐                     │
│         └─────────────────▶│ AlertManager │                     │
│                            │   :9093      │                     │
│                            └──────────────┘                     │
└──────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                         ┌──────────┐
                         │  Client  │
                         │ (cURL,   │
                         │  Python) │
                         └──────────┘
```

## Component Details

### 1. Data Preparation (`src/features/prepare.py`)

**Purpose**: Transform raw data into ML-ready features

**Input**:
- `src/data/raw/default of credit card clients.xls` (30,000 rows, 24 columns)

**Process**:
1. Load Excel data with correct header row
2. Normalize column names (lowercase, underscore-separated)
3. Rename target column: `default_payment_next_month` → `target`
4. Feature engineering:
   - **Utilization ratios**: `util_i = bill_amt_i / limit_bal`
   - **Payment ratios**: `pay_ratio_i = pay_amt_i / bill_amt_i`
   - **Average utilization**: Mean of 6-month utilization
   - **Missed payments**: Count of months with payment delays
5. Train/test split (80/20, stratified)
6. Save processed datasets

**Output**:
- `src/data/processed/train.csv` (24,000 rows, 37 columns)
- `src/data/processed/test.csv` (6,000 rows, 37 columns)
- `src/data/reference/reference.csv` (10,000 rows for monitoring)

**Design Decisions**:
- Stratified split ensures balanced class distribution
- Feature engineering creates domain-specific indicators
- Reference data supports data drift detection

### 2. Model Training (`src/train/train.py`)

**Purpose**: Train ML model and log to MLflow

**Architecture**: scikit-learn Pipeline
```python
Pipeline([
    ("scaler", StandardScaler()),      # Feature normalization
    ("logreg", LogisticRegression())   # Binary classification
])
```

**Training Process**:
1. Load processed train/test data
2. Create pipeline with preprocessing + model
3. Fit pipeline on training data
4. Evaluate on test data
5. Log to MLflow:
   - Parameters: model_type
   - Metrics: accuracy, roc_auc
   - Artifacts: confusion_matrix.json, model

**MLflow Tracking**:
- **Backend Store**: Local file system (`mlruns/`)
- **Artifact Store**: Local file system (`mlruns/`)
- **Experiment**: "credit-card-default"

**Metrics**:
- **Accuracy**: Overall correctness
- **ROC-AUC**: Probability ranking quality
- **Confusion Matrix**: Per-class performance

**Design Decisions**:
- Pipeline ensures preprocessing is included in model
- LogisticRegression chosen for interpretability
- StandardScaler handles feature scale differences

### 3. Model Registration (`src/train/eval_register.py`)

**Purpose**: Register best model to production

**Process**:
1. Search all runs in experiment
2. Rank by `roc_auc` (descending)
3. Select top run
4. Register model to MLflow Model Registry
5. Assign "prod" alias to version

**Model Registry**:
- **Name**: `credit-default-model`
- **Versions**: Incremental (1, 2, 3, ...)
- **Aliases**: `prod`, `staging`, `challenger`

**Design Decisions**:
- Alias-based routing (not version numbers)
- ROC-AUC as primary metric (handles class imbalance)
- Automated selection removes manual errors

### 4. Model Serving (`src/serve/app.py`)

**Purpose**: Serve model predictions via REST API

**FastAPI Application**:

```python
app = FastAPI(
    title="Credit Default Prediction API",
    description="Predicts default probability",
    version="1.0.0"
)
```

**Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/predict` | POST | Make prediction |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc UI |

**Request Flow**:
1. Client sends POST to `/predict` with JSON payload
2. Pydantic validates request schema
3. Feature engineering applied (same as training)
4. Model predicts probability
5. Response includes prediction + metadata

**Feature Engineering**:
- **Must match training**: Same transformations
- **In-request processing**: No pre-computed features
- **Handles missing columns**: Adds ID if needed

**Model Loading**:
```python
MODEL_URI = "models:/credit-default-model@prod"
model = mlflow.pyfunc.load_model(MODEL_URI)
```

**Design Decisions**:
- Load model at startup (not per-request)
- Feature engineering in API ensures consistency
- Prometheus middleware tracks all requests

### 5. MLflow Server

**Purpose**: Centralized experiment tracking and model registry

**Configuration**:
- **Backend Store**: PostgreSQL database
  - Stores metadata (experiments, runs, parameters, metrics)
  - Connection: `postgresql://mlflow:mlflow@postgres:5432/mlflowdb`
- **Artifact Store**: File system
  - Stores artifacts (models, plots, data)
  - Path: `/mlruns`

**Components**:
- **Tracking Server**: Logs experiments
- **Model Registry**: Versions models
- **UI**: Web interface on port 5001

**Design Decisions**:
- PostgreSQL for production reliability
- File artifacts for simplicity (could use S3)
- Separate from API server for scalability

### 6. PostgreSQL Database

**Purpose**: Backend store for MLflow metadata

**Schema** (created by MLflow):
- `experiments`: Experiment metadata
- `runs`: Run information
- `metrics`: Metric values
- `params`: Parameter values
- `tags`: Run tags
- `registered_models`: Model registry
- `model_versions`: Model versions

**Configuration**:
- User: `mlflow`
- Database: `mlflowdb`
- Port: 5432

**Design Decisions**:
- Separate database improves reliability
- Health checks ensure readiness
- Volume mount persists data

### 7. Prometheus

**Purpose**: Collect and store metrics

**Metrics Collected**:

```python
# Counter: Increments only
REQUEST_COUNT = Counter(
    'request_count_total',
    'Total number of requests',
    ['method', 'endpoint']
)

# Histogram: Distribution of values
REQUEST_LATENCY = Histogram(
    'request_latency_seconds',
    'Request latency (seconds)',
    ['endpoint']
)
```

**Configuration** (`prometheus/prometheus.yml`):
```yaml
scrape_configs:
  - job_name: "fastapi"
    static_configs:
      - targets: ["host.docker.internal:8080"]
    scrape_interval: 5s
```

**Query Examples**:
- Request rate: `rate(request_count_total[5m])`
- 95th percentile latency: `histogram_quantile(0.95, request_latency_seconds)`

**Design Decisions**:
- 5-second scrape interval balances freshness/overhead
- Push model avoided (pull is simpler)
- Separate container for isolation

### 8. Grafana

**Purpose**: Visualize metrics from Prometheus

**Features**:
- Pre-built dashboards
- Alert notifications
- Time-series visualization
- Multi-source querying

**Data Source**: Prometheus at `http://prometheus:9090`

**Design Decisions**:
- Separate from Prometheus (SRP)
- Persistent volume for dashboard configs
- Default admin credentials (change in production)

### 9. AlertManager

**Purpose**: Handle alerts from Prometheus

**Configuration** (`prometheus/alertmanager.yml`):
```yaml
route:
  receiver: 'default'
receivers:
  - name: 'default'
    # Add email, Slack, PagerDuty configs
```

**Alert Rules** (`prometheus/prometheus_rules.yml`):
```yaml
groups:
  - name: api_alerts
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, request_latency_seconds) > 1
        for: 5m
```

## Data Flow

### Training Flow

```
Raw Data (.xls)
    │
    ├─▶ prepare.py
    │      │
    │      ├─▶ Feature Engineering
    │      │      │
    │      │      ├─▶ Utilization Ratios
    │      │      ├─▶ Payment Ratios
    │      │      ├─▶ Average Utilization
    │      │      └─▶ Missed Payment Count
    │      │
    │      └─▶ Train/Test Split
    │             │
    │             ├─▶ train.csv
    │             ├─▶ test.csv
    │             └─▶ reference.csv
    │
    └─▶ train.py
           │
           ├─▶ Load Data
           ├─▶ Create Pipeline
           ├─▶ Train Model
           ├─▶ Evaluate
           └─▶ Log to MLflow
                  │
                  ├─▶ Parameters
                  ├─▶ Metrics
                  └─▶ Artifacts
                         │
                         └─▶ eval_register.py
                                │
                                ├─▶ Search Best Run
                                ├─▶ Register Model
                                └─▶ Assign 'prod' Alias
```

### Prediction Flow

```
Client Request (JSON)
    │
    ├─▶ FastAPI Middleware
    │      │
    │      └─▶ Prometheus: request_count++
    │
    ├─▶ Pydantic Validation
    │      │
    │      └─▶ CreditData Schema
    │
    ├─▶ Feature Engineering
    │      │
    │      ├─▶ Lowercase Columns
    │      ├─▶ Add ID Column
    │      ├─▶ Utilization Ratios
    │      ├─▶ Payment Ratios
    │      ├─▶ Average Utilization
    │      └─▶ Missed Payment Count
    │
    ├─▶ Model Prediction
    │      │
    │      ├─▶ Load Model (prod alias)
    │      ├─▶ predict_proba()
    │      └─▶ Threshold at 0.5
    │
    ├─▶ Response Formatting
    │      │
    │      ├─▶ prediction
    │      ├─▶ default_probability
    │      ├─▶ timestamp
    │      └─▶ model_uri
    │
    └─▶ Prometheus: request_latency.observe()
```

### Monitoring Flow

```
FastAPI /metrics Endpoint
    │
    ├─▶ Prometheus Scraper (every 5s)
    │      │
    │      ├─▶ request_count_total
    │      └─▶ request_latency_seconds
    │
    └─▶ Prometheus TSDB
           │
           ├─▶ Grafana Queries
           │      │
           │      └─▶ Visualization Dashboards
           │
           └─▶ Alert Rules Evaluation
                  │
                  └─▶ AlertManager
                         │
                         └─▶ Notifications (Email, Slack)
```

## Model Pipeline

### Pipeline Structure

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, solver="liblinear"))
])
```

### Why Pipeline?

1. **Ensures Preprocessing**: Scaler is part of the model
2. **Prevents Data Leakage**: Fit on train, transform on test
3. **Simplifies Deployment**: Single artifact to load
4. **Guarantees Consistency**: Same transformations at serving

### Feature Transformations

#### Original Features (23)
- LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
- PAY_1 to PAY_6 (6 features)
- BILL_AMT1 to BILL_AMT6 (6 features)
- PAY_AMT1 to PAY_AMT6 (6 features)

#### Engineered Features (13)
- util_1 to util_6 (6 features)
- pay_ratio_1 to pay_ratio_6 (6 features)
- avg_util (1 feature)
- misspay_cnt (1 feature)

#### Total: 37 features (including ID)

### StandardScaler

**Formula**: `z = (x - μ) / σ`

**Learned Parameters**:
- Mean (μ) for each feature
- Standard deviation (σ) for each feature

**Purpose**:
- Normalizes features to mean=0, std=1
- Prevents features with large ranges from dominating
- Required for distance-based algorithms

### LogisticRegression

**Formula**: `P(y=1|x) = 1 / (1 + e^-(w·x + b))`

**Parameters**:
- `max_iter=1000`: Maximum iterations for convergence
- `solver="liblinear"`: Good for small datasets

**Output**:
- Probability between 0 and 1
- Threshold at 0.5 for binary classification

## API Design

### Request Schema (Pydantic)

```python
class CreditData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_1: int
    # ... (23 fields total)
```

**Validation**:
- Type checking (int, float)
- Required fields
- Automatic error messages

### Response Schema

```json
{
  "prediction": 0,                    // Binary (0 or 1)
  "default_probability": 0.1234,      // Float [0, 1]
  "timestamp": "2025-10-28T15:30:00", // ISO 8601
  "model_uri": "models:/credit-default-model@prod"
}
```

### Error Handling

```python
try:
    # Prediction logic
except Exception as e:
    return {
        "error": str(e),
        "message": "Prediction failed",
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat()
    }
```

## Monitoring Stack

### Metrics Hierarchy

```
Application Metrics (FastAPI)
    │
    ├─▶ Request Count
    │      ├─▶ By Method (GET, POST)
    │      └─▶ By Endpoint (/health, /predict)
    │
    └─▶ Request Latency
           ├─▶ Histogram Buckets
           └─▶ By Endpoint
                  │
                  ├─▶ Min, Max, Mean
                  ├─▶ Percentiles (p50, p95, p99)
                  └─▶ Sum, Count
```

### Alert Conditions

```yaml
# High Latency
histogram_quantile(0.95, request_latency_seconds) > 1
# 95th percentile exceeds 1 second

# High Error Rate
rate(request_count_total{status="500"}[5m]) > 0.01
# More than 1% error rate

# Low Request Rate
rate(request_count_total[5m]) < 0.1
# Less than 0.1 requests per second (potential outage)
```

## Infrastructure

### Docker Network

```yaml
networks:
  default:
    driver: bridge
```

**Service Communication**:
- Services communicate via service names (DNS)
- Example: FastAPI → `http://mlflow:5001`

### Volumes

```yaml
volumes:
  postgres_data:    # PostgreSQL data persistence
  grafana_data:     # Grafana dashboards persistence
  ./mlruns:/mlruns  # MLflow artifacts (bind mount)
```

### Health Checks

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U mlflow"]
  interval: 10s
  timeout: 5s
  retries: 5
```

**Purpose**:
- Ensures PostgreSQL is ready before MLflow starts
- Prevents connection errors during startup

## Security Considerations

### Current Implementation (Development)

- Default passwords (admin/admin)
- No authentication on API
- No HTTPS/TLS
- No secrets management
- Internal Docker network only

### Production Recommendations

1. **API Security**:
   ```python
   from fastapi.security import HTTPBearer
   security = HTTPBearer()

   @app.post("/predict")
   def predict(data: CreditData, token: str = Depends(security)):
       # Verify token
   ```

2. **Database Security**:
   - Use strong passwords
   - Store in environment variables
   - Restrict network access

3. **TLS/HTTPS**:
   - Add reverse proxy (Nginx)
   - Use Let's Encrypt certificates

4. **Network Isolation**:
   - Separate networks for frontend/backend
   - Firewall rules

5. **Secrets Management**:
   - Use Docker secrets
   - Or HashiCorp Vault
   - Or AWS Secrets Manager

## Scalability

### Current Limitations

- Single FastAPI instance
- File-based artifact store
- Single PostgreSQL instance
- No load balancing

### Scaling Strategies

#### 1. Horizontal Scaling (FastAPI)

```yaml
services:
  fastapi:
    deploy:
      replicas: 3
```

Add Nginx load balancer:
```nginx
upstream fastapi_backend {
    server fastapi_1:8080;
    server fastapi_2:8080;
    server fastapi_3:8080;
}
```

#### 2. Model Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def get_model():
    return mlflow.pyfunc.load_model(MODEL_URI)

model = get_model()  # Cached globally
```

#### 3. Async Predictions

```python
@app.post("/predict")
async def predict(data: CreditData):
    # Use asyncio for I/O-bound operations
    result = await asyncio.to_thread(model.predict, df)
    return result
```

#### 4. Database Scaling

- PostgreSQL read replicas
- Connection pooling (PgBouncer)
- Managed services (AWS RDS)

#### 5. Artifact Storage

- S3 for artifacts (instead of filesystem)
- CDN for model serving
- Distributed cache (Redis)

#### 6. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi
  template:
    spec:
      containers:
      - name: fastapi
        image: mlops-fastapi:latest
        resources:
          limits:
            cpu: "1"
            memory: "1Gi"
```

## Performance Optimization

### Current Performance

- Model load time: ~2s (startup)
- Prediction latency: ~50-100ms
- Throughput: ~100 requests/sec (single instance)

### Optimization Techniques

1. **Model Loading**:
   - Load once at startup (done)
   - Use model cache

2. **Feature Engineering**:
   - Vectorize operations (done with pandas)
   - Consider numba for numerical ops

3. **API Response**:
   - Use orjson for faster JSON serialization
   - Compress large responses

4. **Database**:
   - Index frequently queried columns
   - Connection pooling

5. **Monitoring**:
   - Reduce scrape interval if needed
   - Aggregate metrics before storage

## Design Trade-offs

### 1. File-based vs Cloud Artifacts

**Choice**: File-based (`./mlruns`)

**Pros**:
- Simple setup
- No cloud dependencies
- Fast local access

**Cons**:
- Not scalable
- No redundancy
- Hard to share

**Alternative**: S3-backed artifacts
```python
mlflow.set_tracking_uri("postgresql://...")
mlflow.set_artifact_uri("s3://bucket/mlruns")
```

### 2. Synchronous vs Asynchronous API

**Choice**: Synchronous with async middleware

**Pros**:
- Simple code
- scikit-learn is synchronous

**Cons**:
- Blocks on I/O

**Alternative**: Full async with separate prediction workers

### 3. Model Pipeline vs Separate Scaler

**Choice**: Pipeline including scaler

**Pros**:
- Single artifact
- Guarantees consistency
- Prevents errors

**Cons**:
- Larger artifact size
- Cannot update scaler independently

### 4. Alias vs Version-based Routing

**Choice**: Alias-based (`@prod`)

**Pros**:
- Decouple deployment from version
- Easy rollbacks

**Cons**:
- Extra abstraction layer

## Future Enhancements

1. **Data Drift Detection**: Evidently integration
2. **Model Performance Monitoring**: Track accuracy over time
3. **A/B Testing**: Traffic splitting between models
4. **Feature Store**: Centralized feature management
5. **CI/CD Pipeline**: Automated testing and deployment
6. **Model Explainability**: SHAP values in responses
7. **Batch Predictions**: Async job queue (Celery)
8. **Model Versioning**: Git-based model code versioning

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/compose.html)
