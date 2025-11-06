# ML Service - CPU/GPU ready container
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 as base-gpu

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    python3-pip curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default and pin pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install all dependencies from requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Verify imports during build (fails build if broken)
RUN python -c "import joblib,catboost,xgboost,sklearn,numpy,pandas,fastapi; print('✅ Build-time imports OK')"

# Copy application code
COPY . /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001

# Default: inference mode
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# CPU-only variant (smaller image)
FROM python:3.11-slim as base-cpu

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Verify imports
RUN python -c "import joblib,catboost,xgboost,sklearn,numpy,pandas,fastapi; print('✅ CPU build-time imports OK')"

# Copy application code
COPY . /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]

# Default to GPU image for production
FROM base-gpu as final