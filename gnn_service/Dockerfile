# ============================================================================
# GNN Service - T-GAT with Self-Supervised Learning
# ============================================================================
FROM hydraulic-ai-base-gpu:cuda12.1

LABEL service="gnn-inference"
LABEL model="t-gat"

WORKDIR /app

# Copy GNN-specific requirements
COPY requirements-gnn.txt .

# Install PyTorch Geometric + dependencies
RUN pip install --no-cache-dir -r requirements-gnn.txt

# Copy GNN service code
COPY . .

# Create directories
RUN mkdir -p /app/models /app/data /app/checkpoints

# GPU environment
ENV CUDA_VISIBLE_DEVICES=0

# Healthcheck
HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# Run
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
