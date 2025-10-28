# Credit Card Default Prediction - MLOps Pipeline

A complete MLOps pipeline for predicting credit card default using machine learning with MLflow, FastAPI, Prometheus, and Grafana.

## Overview

This project demonstrates an end-to-end MLOps workflow including:
- Data preprocessing and feature engineering
- Model training with MLflow experiment tracking
- Model registry and versioning
- REST API serving with FastAPI
- Monitoring with Prometheus and Grafana
- Containerization with Docker

## Project Structure

```
mlops/
├── src/
│   ├── data/
│   │   ├── raw/                    # Raw dataset
│   │   ├── processed/              # Train/test splits
│   │   └── reference/              # Reference data for monitoring
│   ├── features/
│   │   └── prepare.py              # Data preprocessing & feature engineering
│   ├── train/
│   │   ├── train.py                # Model training with MLflow
│   │   └── eval_register.py        # Model evaluation & registration
│   └── serve/
│       └── app.py                  # FastAPI serving application
├── mlruns/                         # MLflow tracking data
├── prometheus/
│   ├── prometheus.yml              # Prometheus configuration
│   ├── prometheus_rules.yml        # Alerting rules
│   └── alertmanager.yml            # Alert manager config
├── docker-compose.yml              # Multi-container orchestration
├── Dockerfile                      # FastAPI app container
├── Dockerfile.mlflow              # MLflow server container
├── requirements.txt               # Python dependencies
└── test_request.json              # Sample prediction request
```

## Key Features

### Machine Learning
- **Dataset**: Credit card default dataset with 23 features
- **Model**: Logistic Regression with StandardScaler pipeline
- **Feature Engineering**: Utilization ratios, payment ratios, average utilization, missed payment counts
- **Metrics**: Accuracy, ROC-AUC, Confusion Matrix

### MLOps Components
- **MLflow**: Experiment tracking, model registry, and versioning
- **FastAPI**: Production-ready REST API with automatic documentation
- **Prometheus**: Metrics collection (request count, latency)
- **Grafana**: Monitoring dashboards
- **Docker**: Containerized deployment with PostgreSQL backend for MLflow

## Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- At least 4GB RAM available for containers

### Installation

1. **Clone and navigate to the project**
   ```bash
   cd /Users/aashigoyani/Downloads/mlops
   ```

2. **Create virtual environment and install dependencies**
   ```bash
   python3 -m venv mlops
   source mlops/bin/activate  # On Windows: mlops\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Prepare the data**
   ```bash
   python src/features/prepare.py
   ```

4. **Train the model**
   ```bash
   python src/train/train.py
   ```

5. **Register the best model**
   ```bash
   python src/train/eval_register.py
   ```

### Running with Docker (Recommended)

1. **Start all services**
   ```bash
   docker-compose up -d
   ```

   This starts:
   - PostgreSQL (port 5432)
   - MLflow UI (port 5001)
   - FastAPI (port 8080)
   - Prometheus (port 9090)
   - Grafana (port 3000)
   - AlertManager (port 9093)

2. **Check service health**
   ```bash
   docker-compose ps
   ```

3. **View logs**
   ```bash
   docker-compose logs -f fastapi
   ```

### Making Predictions

**Health check:**
```bash
curl http://localhost:8080/health
```

**Make a prediction:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

**Response:**
```json
{
  "prediction": 0,
  "default_probability": 0.1234,
  "timestamp": "2025-10-28T15:30:00",
  "model_uri": "models:/credit-default-model@prod"
}
```

## Accessing Services

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI API | http://localhost:8080 | - |
| FastAPI Docs | http://localhost:8080/docs | - |
| MLflow UI | http://localhost:5001 | - |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin/admin |
| AlertManager | http://localhost:9093 | - |

## API Endpoints

### GET /health
Health check endpoint
```bash
curl http://localhost:8080/health
```

### POST /predict
Predict credit default probability
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

### GET /metrics
Prometheus metrics endpoint
```bash
curl http://localhost:8080/metrics
```

## Monitoring

### Prometheus Metrics
- `request_count_total`: Total number of requests by method and endpoint
- `request_latency_seconds`: Request latency histogram by endpoint

### Grafana Dashboards
1. Login to Grafana (http://localhost:3000)
2. Default credentials: admin/admin
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboards from `mlops/grafana/dashboards/`

## Development Workflow

1. **Data Preparation**
   - Place raw data in `src/data/raw/`
   - Run `python src/features/prepare.py`
   - Generates train/test/reference datasets

2. **Model Training**
   - Run `python src/train/train.py`
   - Logs experiments to MLflow
   - Creates model artifacts

3. **Model Registration**
   - Run `python src/train/eval_register.py`
   - Registers best model to MLflow registry
   - Assigns "prod" alias to best version

4. **Local Testing**
   - Start FastAPI: `uvicorn src.serve.app:app --reload`
   - Test predictions: Use test_request.json

5. **Production Deployment**
   - Build and deploy with `docker-compose up -d`
   - Monitor via Grafana/Prometheus

## Troubleshooting

### Container Issues
```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker-compose logs -f [service-name]
```

### Model Loading Issues
- Ensure model is registered: Check MLflow UI at http://localhost:5001
- Verify "prod" alias exists
- Check mlruns directory permissions

### Database Connection Issues
```bash
# Check PostgreSQL health
docker-compose exec postgres pg_isready -U mlflow
```

## Technology Stack

- **Python 3.9**: Core language
- **scikit-learn**: Machine learning
- **MLflow 2.14**: Experiment tracking & model registry
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Docker**: Containerization
- **PostgreSQL**: MLflow backend store

## Model Details

### Features (23 total)
- `LIMIT_BAL`: Credit limit
- `SEX`: Gender (1=male, 2=female)
- `EDUCATION`: Education level
- `MARRIAGE`: Marital status
- `AGE`: Age in years
- `PAY_1` to `PAY_6`: Payment status for past 6 months
- `BILL_AMT1` to `BILL_AMT6`: Bill amounts for past 6 months
- `PAY_AMT1` to `PAY_AMT6`: Payment amounts for past 6 months

### Engineered Features
- `util_1` to `util_6`: Utilization ratios (bill/limit)
- `pay_ratio_1` to `pay_ratio_6`: Payment ratios (payment/bill)
- `avg_util`: Average utilization across 6 months
- `misspay_cnt`: Count of months with delayed payments

### Model Pipeline
1. **StandardScaler**: Normalize features
2. **LogisticRegression**: Binary classification (max_iter=1000, solver=liblinear)

