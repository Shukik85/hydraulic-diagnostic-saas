# ML Service - CPU/GPU ready container
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as base-gpu

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    python3-pip curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Install ML dependencies
RUN pip install --upgrade pip setuptools wheel

# Core ML packages with GPU support
RUN pip install \
    catboost==1.2.7 \
    xgboost==2.0.3 \
    scikit-learn==1.5.2 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    joblib==1.4.2

# FastAPI and async
RUN pip install \
    fastapi==0.115.4 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.9.2 \
    aiohttp==3.10.10

# Observability
RUN pip install \
    structlog==24.4.0 \
    rich==13.9.2 \
    prometheus-client==0.21.0

# Production
RUN pip install \
    gunicorn==23.0.0 \
    python-multipart==0.0.12

WORKDIR /app

# Copy ML service code
COPY . .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001

# Default: inference mode
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# CPU-only variant (smaller image)
FROM python:3.11-slim as base-cpu

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install \
    catboost==1.2.7 \
    xgboost==2.0.3 \
    scikit-learn==1.5.2 \
    numpy==1.26.4 \
    pandas==2.2.3 \
    joblib==1.4.2 \
    fastapi==0.115.4 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.9.2 \
    aiohttp==3.10.10 \
    structlog==24.4.0 \
    rich==13.9.2 \
    prometheus-client==0.21.0 \
    gunicorn==23.0.0 \
    python-multipart==0.0.12

WORKDIR /app
COPY . .

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# Default to GPU image for production
FROM base-gpu as final