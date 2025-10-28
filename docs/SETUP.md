# Setup Guide

This guide walks you through setting up the Credit Card Default Prediction MLOps pipeline from scratch.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Docker Setup](#docker-setup)
- [Data Setup](#data-setup)
- [Running the Pipeline](#running-the-pipeline)
- [Verification](#verification)
- [Common Issues](#common-issues)

## Prerequisites

### Required Software
- **Python 3.9+**: [Download](https://www.python.org/downloads/)
- **Docker**: [Download](https://www.docker.com/get-started)
- **Docker Compose**: Included with Docker Desktop
- **Git**: [Download](https://git-scm.com/downloads)

### System Requirements
- **RAM**: Minimum 4GB available (8GB recommended)
- **Disk Space**: 5GB free space
- **OS**: macOS, Linux, or Windows with WSL2

### Verify Installation
```bash
python3 --version    # Should be 3.9 or higher
docker --version     # Should be 20.10 or higher
docker-compose --version
```

## Local Development Setup

### 1. Clone the Repository
```bash
cd /path/to/your/workspace
# If you have the project already, navigate to it
cd /Users/aashigoyani/Downloads/mlops
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv mlops

# Activate virtual environment
# On macOS/Linux:
source mlops/bin/activate

# On Windows:
mlops\Scripts\activate

# Verify activation (should show path to venv)
which python
```

### 3. Install Python Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Verify installation
pip list
```

Expected packages:
- pandas
- numpy
- scikit-learn
- mlflow
- fastapi
- uvicorn
- pydantic
- prometheus-client
- evidently
- python-dotenv
- requests

### 4. Verify Package Installation
```bash
python -c "import mlflow; print(f'MLflow version: {mlflow.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')"
```

## Docker Setup

### 1. Verify Docker is Running
```bash
docker ps
# Should show a list of running containers (can be empty)
```

### 2. Build Docker Images
```bash
# Build all images defined in docker-compose.yml
docker-compose build

# Or build specific services
docker-compose build fastapi
docker-compose build mlflow
```

### 3. Pull Required Images
```bash
# Pull base images
docker-compose pull postgres
docker-compose pull prometheus
docker-compose pull grafana
docker-compose pull alertmanager
```

## Data Setup

### 1. Verify Raw Data Exists
```bash
# Check if the raw data file exists
ls -lh src/data/raw/

# Should show: default of credit card clients.xls
```

If the file doesn't exist:
- Download the [UCI Credit Card Default dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- Place the `.xls` file in `src/data/raw/`

### 2. Run Data Preparation
```bash
# This creates processed datasets
python src/features/prepare.py
```

**Expected output:**
```
📂 Loading dataset...
✅ Loaded dataset with shape: (30000, 24)
🧮 Engineering features...
📊 Splitting data into train/test...
✅ Data preparation complete!
 - src/data/processed/train.csv
 - src/data/processed/test.csv
 - src/data/reference/reference.csv
```

### 3. Verify Processed Data
```bash
ls -lh src/data/processed/
# Should show: train.csv, test.csv

ls -lh src/data/reference/
# Should show: reference.csv
```

## Running the Pipeline

### Option A: Local Development (Without Docker)

#### 1. Start MLflow Server
```bash
# Terminal 1: Start MLflow UI
mlflow ui --backend-store-uri file://$(pwd)/mlruns --port 5001
```

#### 2. Train Model
```bash
# Terminal 2: Run training
python src/train/train.py
```

**Expected output:**
```
✅ MLflow using local tracking
📦 Loading processed datasets...
✅ Pipeline logged | Accuracy=0.8192 | AUC=0.7745
🎯 Training complete! Check MLflow UI for details:
   → http://localhost:5001
```

#### 3. Register Best Model
```bash
python src/train/eval_register.py
```

**Expected output:**
```
🔍 Fetching experiment runs...
🏆 Best run found: xxx | ROC_AUC=0.7745
✅ Model 'credit-default-model' registered and aliased as 'prod' (v1)
```

#### 4. Start FastAPI Server
```bash
# Terminal 3: Start API server
uvicorn src.serve.app:app --host 0.0.0.0 --port 8080
```

**Expected output:**
```
🔄 Loading Production model from MLflow: models:/credit-default-model@prod
✅ Pipeline model (scaler + logistic regression) loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8080
```

### Option B: Docker Deployment (Recommended for Production)

#### 1. Start All Services
```bash
docker-compose up -d
```

**Expected output:**
```
Creating network "mlops_default" with the default driver
Creating volume "mlops_postgres_data" with default driver
Creating volume "mlops_grafana_data" with default driver
Creating mlflow_postgres ... done
Creating mlflow_server   ... done
Creating fastapi_model_server ... done
Creating prometheus      ... done
Creating alertmanager    ... done
Creating grafana        ... done
```

#### 2. Check Service Status
```bash
docker-compose ps
```

**Expected output:**
```
NAME                    STATUS              PORTS
mlflow_postgres        Up (healthy)        5432/tcp
mlflow_server          Up                  5001/tcp
fastapi_model_server   Up                  8080/tcp
prometheus             Up                  9090/tcp
grafana                Up                  3000/tcp
alertmanager          Up                  9093/tcp
```

#### 3. View Logs
```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f fastapi
docker-compose logs -f mlflow
docker-compose logs -f postgres
```

#### 4. Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes data)
docker-compose down -v
```

## Verification

### 1. Verify MLflow
```bash
# Open browser to http://localhost:5001
# You should see:
# - Experiment: "credit-card-default"
# - Runs with metrics (accuracy, roc_auc)
# - Registered model: "credit-default-model"
```

### 2. Verify FastAPI
```bash
# Health check
curl http://localhost:8080/health

# API documentation
# Open browser to http://localhost:8080/docs

# Test prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### 3. Verify Prometheus
```bash
# Open browser to http://localhost:9090

# Query examples:
# - request_count_total
# - request_latency_seconds

# Check targets: http://localhost:9090/targets
# FastAPI target should be "UP"
```

### 4. Verify Grafana
```bash
# Open browser to http://localhost:3000
# Login: admin / admin
# Add Prometheus data source:
#   - URL: http://prometheus:9090
#   - Click "Save & Test"
```

### 5. Run End-to-End Test
```bash
# Test script
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
  }' | python -m json.tool
```

**Expected response:**
```json
{
  "prediction": 0,
  "default_probability": 0.1234,
  "timestamp": "2025-10-28T15:30:00",
  "model_uri": "models:/credit-default-model@prod"
}
```

## Common Issues

### Issue 1: Port Already in Use
```bash
# Error: port 8080 is already allocated
# Solution: Kill the process using the port
lsof -ti:8080 | xargs kill -9

# Or change port in docker-compose.yml
# Change "8080:8080" to "8081:8080"
```

### Issue 2: MLflow Cannot Find Model
```bash
# Error: Model 'credit-default-model' not found
# Solution: Ensure you've registered the model
python src/train/eval_register.py

# Verify in MLflow UI: http://localhost:5001
```

### Issue 3: Docker Container Fails to Start
```bash
# Check logs
docker-compose logs [service-name]

# Restart specific service
docker-compose restart [service-name]

# Rebuild and restart
docker-compose down
docker-compose build --no-cache [service-name]
docker-compose up -d
```

### Issue 4: PostgreSQL Connection Issues
```bash
# Wait for PostgreSQL to be ready
docker-compose exec postgres pg_isready -U mlflow

# Check PostgreSQL logs
docker-compose logs postgres

# Restart PostgreSQL
docker-compose restart postgres
```

### Issue 5: Permission Denied on mlruns/
```bash
# Fix permissions
chmod -R 755 mlruns/
chmod -R 755 src/data/
```

### Issue 6: Python Module Not Found
```bash
# Ensure virtual environment is activated
source mlops/bin/activate  # On macOS/Linux
mlops\Scripts\activate     # On Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue 7: Data File Not Found
```bash
# Verify raw data exists
ls -lh src/data/raw/

# Run data preparation
python src/features/prepare.py
```

## Next Steps

After successful setup:
1. Read [USAGE.md](USAGE.md) for detailed usage examples
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) for system architecture
3. Explore the MLflow UI to understand experiments
4. Test the API using the Swagger docs at http://localhost:8080/docs
5. Configure Grafana dashboards for monitoring

## Environment Variables

Create a `.env` file (optional):
```bash
# MLflow
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_BACKEND_STORE_URI=postgresql://mlflow:mlflow@localhost:5432/mlflowdb

# FastAPI
API_HOST=0.0.0.0
API_PORT=8080

# Prometheus
PROMETHEUS_PORT=9090

# Grafana
GRAFANA_PORT=3000
GF_SECURITY_ADMIN_USER=admin
GF_SECURITY_ADMIN_PASSWORD=admin
```

## Cleanup

### Remove All Containers and Volumes
```bash
# Stop and remove containers
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker rmi $(docker images -q mlops*)
```

### Clean Python Cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

### Deactivate Virtual Environment
```bash
deactivate
```

## Getting Help

- Check the [main README](../README.md)
- Review [USAGE.md](USAGE.md) for examples
- Check Docker logs: `docker-compose logs -f`
- Check MLflow UI: http://localhost:5001
- Check FastAPI docs: http://localhost:8080/docs
