# services/rag_service/Dockerfile
# Production Dockerfile для DeepSeek-R1-Distill-32B на 2x A100
# UPDATED: Fixed Python path configuration and removed unused dependencies

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# FIXED: Set PYTHONPATH to fix import issues
ENV PYTHONPATH=/app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install vLLM для эффективного inference
RUN pip install --no-cache-dir \
    vllm==0.7.3 \
    torch==2.5.1 \
    transformers==4.46.0 \
    accelerate==0.35.0 \
    bitsandbytes==0.44.1

# Install FastAPI и supporting libraries
RUN pip install --no-cache-dir \
    fastapi==0.115.0 \
    uvicorn[standard]==0.32.0 \
    pydantic==2.9.2 \
    pydantic-settings==2.6.0

# REMOVED: ChromaDB и LlamaIndex (not used in vLLM implementation)
# Saves ~500MB of image size

# Install monitoring
RUN pip install --no-cache-dir \
    prometheus-client==0.21.0 \
    structlog==24.4.0 \
    psutil==6.1.0 \
    pynvml==12.560.30

# Copy application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories
RUN mkdir -p /app/models /app/config /app/logs

# Download model (optional, можно mount volume)
# RUN python download_model.py

EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8004/health', timeout=5)"

# Production command with vLLM
# Uses main.py from root (not app/main.py)
CMD ["python", "main.py"]
