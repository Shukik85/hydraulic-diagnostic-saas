# GPU-enabled Dockerfile for GNN Service
# Optimized for GTX 1650 SUPER (4GB VRAM) with CUDA 12.4
# Compatible with CUDA 12.9 driver

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 gnn && chown -R gnn:gnn /app
USER gnn

# GPU Environment Variables (optimized for 4GB VRAM)
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import requests; assert torch.cuda.is_available(); requests.get('http://localhost:8001/health/')" || exit 1

# Expose port
EXPOSE 8001

# Run application
CMD ["uvicorn", "inference_service:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1"]
