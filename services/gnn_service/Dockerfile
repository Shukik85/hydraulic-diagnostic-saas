# Production Dockerfile: Python 3.13 + PyTorch 2.8 + CUDA 12.9
# Optimized for GTX 1650 (4GB VRAM)
# Universal Temporal GNN + RAG architecture

FROM nvidia/cuda:12.6.1-cudnn9-runtime-ubuntu24.04

# Install Python 3.13
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    python3-pip \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Install pip for Python 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

WORKDIR /app

# Install PyTorch 2.8 + CUDA 12.9 first
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Install torch-geometric ecosystem (prebuilt wheels)
RUN pip install --no-cache-dir torch-geometric && \
    pip install --no-cache-dir \
    https://data.pyg.org/whl/torch-2.8.0%2Bcu129/torch_scatter-2.1.2%2Bpt28cu129-cp313-cp313-linux_x86_64.whl \
    https://data.pyg.org/whl/torch-2.8.0%2Bcu129/torch_sparse-0.6.18%2Bpt28cu129-cp313-cp313-linux_x86_64.whl

# Copy requirements (without torch/pyg)
COPY requirements.txt .

# Install remaining dependencies
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic asyncpg numpy pandas \
    python-dotenv openai anthropic redis prometheus-client python-json-logger pytest

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 gnn && chown -R gnn:gnn /app
USER gnn

# GPU optimizations for GTX 1650 (4GB VRAM)
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
ENV TORCH_CUDA_ARCH_LIST="7.5"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV OMP_NUM_THREADS=4
ENV CUDA_LAUNCH_BLOCKING=0

# Python optimizations for 3.13
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONOPTIMIZE=2

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" && \
        curl -f http://localhost:8001/health || exit 1

EXPOSE 8001

# Run FastAPI with optimized workers
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "info"]
