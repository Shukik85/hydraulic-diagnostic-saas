# services/rag_service/Dockerfile
# Production Dockerfile для DeepSeek-R1-Distill-32B на 2x A100

FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

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
    vllm==0.6.0 \
    torch==2.1.2 \
    transformers==4.36.0 \
    accelerate==0.25.0 \
    bitsandbytes==0.41.3

# Install FastAPI и gRPC
RUN pip install --no-cache-dir \
    fastapi==0.108.0 \
    uvicorn[standard]==0.24.0 \
    grpcio==1.60.0 \
    grpcio-tools==1.60.0 \
    protobuf==4.25.0

# Install ChromaDB и LlamaIndex
RUN pip install --no-cache-dir \
    chromadb==0.4.22 \
    llama-index==0.9.48 \
    sentence-transformers==2.2.2

# Install monitoring
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    structlog==24.1.0

# Copy application
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create directories
RUN mkdir -p /app/models /app/knowledge_base /app/logs

# Download model (optional, можно mount volume)
# RUN python download_model.py

EXPOSE 8004 50052

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8004/health', timeout=5)"

# Production command with vLLM
CMD ["python", "main.py"]
