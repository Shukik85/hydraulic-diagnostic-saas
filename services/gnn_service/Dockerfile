# Production Dockerfile for GNN Service
# Python 3.14.0 + PyTorch 2.8.0 + CUDA 12.8
# Optimized multi-stage build for minimal image size

# ==================== Stage 1: Base ====================
FROM nvidia/cuda:12.8.0-cudnn9-runtime-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install Python 3.14 and minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.14 \
        python3.14-dev \
        python3.14-distutils \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.14 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.14 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.14 1

# Install pip for Python 3.14
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14
RUN python3.14 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ==================== Stage 2: Dependencies ====================
FROM base AS dependencies

WORKDIR /tmp

# Copy requirements
COPY requirements.txt .

# Install PyTorch 2.8.0 with CUDA 12.8 support first (separate layer for caching)
RUN pip install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.19.0 \
    torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==================== Stage 3: Application ====================
FROM base AS application

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code with proper ownership
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p \
    logs \
    models/checkpoints \
    models/production \
    data/raw \
    data/processed \
    checkpoints/dev \
    checkpoints/production \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Expose port
EXPOSE 8002

# Start application with uvicorn
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "2", \
     "--log-level", "info"]
