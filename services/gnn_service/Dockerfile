FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch Geometric dependencies
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torch-geometric==2.4.0 \
    torch-scatter \
    torch-sparse

# Copy application
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data && \
    chmod 755 /app/models /app/data

# Create non-root user
RUN useradd -m -u 1000 gnnuser && \
    chown -R gnnuser:gnnuser /app
USER gnnuser

EXPOSE 8001

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/gnn/health || exit 1

CMD ["uvicorn", "inference_service:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
