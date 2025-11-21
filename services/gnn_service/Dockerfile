# Production Dockerfile for GNN Service
# Python 3.13.5 + PyTorch 2.8 + CUDA 12.9

# Stage 1: Base image with CUDA
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 AS base

# Python 3.13 installation
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Stage 2: Dependencies
FROM base AS dependencies

WORKDIR /tmp

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM base AS application

# Copy installed packages from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Create app user (non-root)
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Create necessary directories
RUN mkdir -p logs models/checkpoints models/production data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Expose port
EXPOSE 8002

# Start application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8002", "--workers", "2"]