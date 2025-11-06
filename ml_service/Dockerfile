# ML Service - CPU/GPU ready container with venv isolation
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base-gpu

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3.11-venv \
    python3-pip curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment
ENV VENV_PATH=/opt/venv
RUN python3.11 -m venv ${VENV_PATH}

# Use venv python/pip by default
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Upgrade pip inside venv
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Install all dependencies in venv
RUN pip install --no-cache-dir -r /app/requirements.txt

# Verify imports during build (fails build if broken)
RUN python -c "import sys; print('Python executable:', sys.executable); import joblib,catboost,xgboost,sklearn,numpy,pandas,fastapi; print('✅ Build-time imports OK')"

# Copy application code
COPY . /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001

# Default: inference mode
CMD ["python", "-m", "uvicorn", "simple_predict:app", "--host", "0.0.0.0", "--port", "8001"]

# CPU-only variant with venv
FROM python:3.11-slim AS base-cpu

RUN apt-get update && apt-get install -y curl build-essential && rm -rf /var/lib/apt/lists/*

# Create venv for CPU variant too
ENV VENV_PATH=/opt/venv
RUN python -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Verify CPU imports
RUN python -c "import sys; print('CPU Python executable:', sys.executable); import joblib,catboost,xgboost,sklearn,numpy,pandas,fastapi; print('✅ CPU build-time imports OK')"

COPY . /app

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/healthz || exit 1

EXPOSE 8001
CMD ["python", "-m", "uvicorn", "simple_predict:app", "--host", "0.0.0.0", "--port", "8001"]

# Default to GPU image for production
FROM base-gpu AS final