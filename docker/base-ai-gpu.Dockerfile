# ============================================================================
# Base AI GPU Image - CUDA 12.1 with pre-downloaded PyTorch wheels
# ============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

LABEL maintainer="Aleksandr Plotnikov <a.s.plotnikov85@gmail.com>"
LABEL description="Shared GPU base for GNN and RAG services"
LABEL cuda.version="12.1.1"

# Environment
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    curl git build-essential \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Copy pre-downloaded PyTorch wheels (FIXED PATH!)
COPY docker/wheels/*.whl /tmp/wheels/

# Install PyTorch from local wheels (no download!)
RUN pip install --no-cache-dir \
    /tmp/wheels/torch-2.3.1+cu121-cp311-cp311-linux_x86_64.whl \
    /tmp/wheels/torchvision-0.18.1+cu121-cp311-cp311-linux_x86_64.whl \
    && rm -rf /tmp/wheels

# Install common libraries
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.11.4 \
    scikit-learn==1.5.2 \
    fastapi==0.115.4 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.9.2 \
    pydantic-settings==2.6.0 \
    httpx==0.27.0 \
    prometheus-client==0.21.0 \
    structlog==24.4.0
