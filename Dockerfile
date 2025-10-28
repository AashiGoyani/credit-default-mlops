# ========================
# FastAPI + MLflow + Prometheus app
# ========================

FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . /app

# Install system dependencies (optional)
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8080

# Run the FastAPI server
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "8080"]
