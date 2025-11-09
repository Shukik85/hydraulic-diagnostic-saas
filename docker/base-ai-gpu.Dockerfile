# ============================================================================
# Base AI GPU Image - CUDA 12.4 + PyTorch for ML & RAG services
# ============================================================================
FROM nvidia/cuda:12.4.0-cudnn-runtime-ubuntu22.04

# Metadata
LABEL maintainer="Aleksandr Plotnikov <a.s.plotnikov85@gmail.com>"
LABEL description="Shared GPU base image for ML and RAG services"
LABEL cuda.version="12.4"

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    curl \
    git \
    build-essential \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install common AI/ML libraries
RUN pip install --no-cache-dir \
    # Core ML
    numpy==1.26.4 \
    scipy==1.11.4 \
    scikit-learn==1.5.2 \
    pandas==2.2.3 \
    # FastAPI Stack
    fastapi==0.115.4 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.9.2 \
    pydantic-settings==2.6.0 \
    # Async & HTTP
    httpx==0.27.0 \
    aiohttp==3.10.10 \
    # Monitoring
    prometheus-client==0.21.0 \
    structlog==24.4.0 \
    # NLP Base
    transformers==4.44.0 \
    sentence-transformers==3.0.1

# Cleanup
RUN pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Healthcheck utility
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*
