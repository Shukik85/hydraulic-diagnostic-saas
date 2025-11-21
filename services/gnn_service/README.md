# GNN Service - Production-Ready Implementation

🌱 **Status:** In Active Development  
🔖 **Branch:** `feature/gnn-service-production-ready`  
📅 **Created:** 2025-11-21

---

## 🚀 Overview

Production-ready Graph Neural Network service for hydraulic system diagnostics using **Universal Temporal GNN** (GAT + LSTM architecture).

### Key Features

- ✅ **Clean Architecture** - Modular `src/` organization, no stub files
- 🧠 **Universal Temporal GNN** - GAT (Graph Attention) + LSTM for time-series
- ⚡ **Modern Stack** - Python 3.13.5, PyTorch 2.8, CUDA 12.9
- 🚀 **FastAPI** - Async/await, Pydantic v2 validation
- 📊 **Observability** - Structured logging, Prometheus metrics
- 🔄 **Production Pipeline** - PyTorch Lightning, distributed training
- 🐳 **Containerized** - Docker with CUDA support

---

## 📚 Quick Start

### Prerequisites

- **Python 3.13.5**
- **CUDA 12.9** (for GPU support)
- **Docker** (optional)
- **TimescaleDB** (for sensor data)

### Installation

```bash
# Clone repository
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas
git checkout feature/gnn-service-production-ready

# Navigate to service
cd services/gnn_service

# Create virtual environment
python3.13 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Set up environment
cp .env.example .env
# Edit .env with your configuration

# Run service
uvicorn api.main:app --reload --port 8002
```

### Docker Development

```bash
# Build dev image
docker build -f Dockerfile.dev -t gnn-service:dev .

# Run with hot reload
docker run -p 8002:8002 \
  -v $(pwd):/app \
  --gpus all \
  gnn-service:dev
```

### Docker Production

```bash
# Build production image
docker build -t gnn-service:latest .

# Run production
docker run -p 8002:8002 \
  --gpus all \
  -e DATABASE_URL=postgresql://... \
  gnn-service:latest
```

---

## 🏛️ Architecture

### Directory Structure

```
services/gnn_service/
├── src/                      # Source code
│   ├── models/              # GNN models
│   ├── data/                # Data processing
│   ├── inference/           # Inference engine
│   ├── training/            # Training pipeline
│   ├── schemas/             # Pydantic models
│   └── utils/               # Utilities
├── api/                     # FastAPI application
│   ├── main.py
│   ├── routes/
│   └── middleware/
├── config/                  # Configuration
├── tests/                   # Tests
├── _legacy/                 # Archived old code
├── data/                    # Data directory
├── models/                  # Saved models
├── logs/                    # Logs
└── kubernetes/              # K8s manifests
```

See [STRUCTURE.md](STRUCTURE.md) for detailed architecture.

### GNN Model Architecture

```
Input: Sensor Time-Series
        ↓
   Preprocessing
        ↓
   Graph Construction (Dynamic)
        ↓
   GAT Layers (×3) → Spatial Attention
        ↓
   LSTM Layers (×2) → Temporal Modeling
        ↓
   Output Heads:
     - Health Score (0-1)
     - Degradation Rate
     - Anomaly Detection
```

---

## 💻 API Usage

### Health Check

```bash
curl http://localhost:8002/health
```

**Response:**
```json
{
  "service": "gnn-service",
  "status": "healthy",
  "timestamp": "2025-11-21T00:15:00Z",
  "checks": {
    "model": "loaded",
    "database": "connected",
    "gpu": "available"
  }
}
```

### Inference

```bash
curl -X POST http://localhost:8002/inference \
  -H "Content-Type: application/json" \
  -d '{
    "equipment_id": "excavator_001",
    "time_window": {
      "start_time": "2025-11-01T00:00:00Z",
      "end_time": "2025-11-13T00:00:00Z"
    }
  }'
```

**Response:**
```json
{
  "request_id": "req_1732147500000",
  "overall_health_score": 0.87,
  "component_health": [
    {
      "component_id": "pump_main",
      "component_type": "hydraulic_pump",
      "health_score": 0.92,
      "degradation_rate": 0.02,
      "confidence": 0.95
    }
  ],
  "anomalies": [
    {
      "anomaly_type": "pressure_drop",
      "severity": "medium",
      "confidence": 0.85,
      "affected_components": ["valve_01"],
      "description": "Unusual pressure fluctuation detected"
    }
  ],
  "recommendations": [
    "Schedule maintenance for valve_01 within 7 days",
    "Monitor pump_main pressure levels"
  ],
  "inference_time_ms": 387.5,
  "timestamp": "2025-11-21T00:15:00Z",
  "model_version": "2.0.0"
}
```

### Batch Inference

```bash
curl -X POST http://localhost:8002/batch-inference \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"equipment_id": "exc_001", "time_window": {...}},
      {"equipment_id": "exc_002", "time_window": {...}}
    ]
  }'
```

### API Documentation

Interactive API docs:
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc
- **OpenAPI JSON**: http://localhost:8002/openapi.json

---

## 🎯 Training

### Dataset Preparation

```bash
# Prepare data
python -m src.data.preprocessing \
  --input data/raw/sensor_data.csv \
  --output data/processed/ \
  --metadata data/metadata/equipment.json
```

### Train Model

```bash
# Single GPU
python -m src.training.train \
  --config config/training.yaml \
  --data data/processed/ \
  --output models/checkpoints/

# Multi-GPU (DDP)
python -m src.training.train \
  --config config/training.yaml \
  --data data/processed/ \
  --output models/checkpoints/ \
  --gpus 4 \
  --strategy ddp
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard

# Weights & Biases (optional)
wandb login
python -m src.training.train --wandb
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Specific test
pytest tests/unit/test_models.py::test_gnn_forward
```

### Code Quality

```bash
# Format code
black src/ api/ tests/
isort src/ api/ tests/

# Lint
ruff check src/ api/ tests/

# Type check
mypy src/ api/

# Security scan
bandit -r src/ api/
```

---

## 📊 Monitoring

### Metrics Endpoint

```bash
curl http://localhost:8002/metrics
```

**Key Metrics:**
- `gnn_inference_duration_seconds` - Inference latency histogram
- `gnn_inference_total` - Total inference requests
- `gnn_inference_errors_total` - Error count
- `gnn_model_load_time_seconds` - Model loading time
- `gnn_gpu_utilization_percent` - GPU utilization
- `gnn_batch_size` - Batch size distribution

### Health Checks

```bash
# Liveness probe
curl http://localhost:8002/health

# Readiness probe
curl http://localhost:8002/ready
```

### Logs

```bash
# View logs
tail -f logs/gnn-service.log

# Structured JSON logs
cat logs/gnn-service.log | jq '.'
```

---

## 🚀 Deployment

### Kubernetes

```bash
# Apply manifests
kubectl apply -f kubernetes/

# Check pods
kubectl get pods -n hydraulic-prod

# View logs
kubectl logs -f deployment/gnn-service -n hydraulic-prod

# Port forward
kubectl port-forward svc/gnn-service 8002:8002 -n hydraulic-prod
```

### Environment Variables

```bash
# Required
DATABASE_URL=postgresql://user:pass@host:5432/db
REDIS_URL=redis://host:6379/0
MODEL_PATH=/app/models/production/model.ckpt
METADATA_PATH=/app/data/metadata/equipment.json

# Optional
LOG_LEVEL=INFO
WORKERS=2
BATCH_SIZE=32
GPU_DEVICE=cuda:0
ENABLE_METRICS=true
```

---

## 📝 Documentation

- **[Roadmap](../../docs/GNN_SERVICE_ROADMAP.md)** - Implementation plan
- **[Structure](STRUCTURE.md)** - Architecture details
- **[API Reference](http://localhost:8002/docs)** - OpenAPI documentation
- **[Training Guide](docs/TRAINING.md)** - Model training
- **[Deployment Guide](../../docs/DEPLOYMENT.md)** - Production deployment

---

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

### Development Workflow

1. Create feature branch from `feature/gnn-service-production-ready`
2. Make changes following code style guidelines
3. Write tests (coverage ≥ 80%)
4. Run code quality checks
5. Submit pull request

### Code Style

- **Formatter**: Black (line length 88)
- **Linter**: Ruff
- **Type checker**: MyPy (strict mode)
- **Docstrings**: Google style

---

## ⚠️ Known Issues

- [ ] PyTorch 2.8 not yet released (using 2.2.0)
- [ ] Python 3.13.5 compatibility testing ongoing
- [ ] CUDA 12.9 optimization in progress

---

## 📜 License

Proprietary - Hydraulic Diagnostic SaaS

---

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues)
- **Email**: support@hydraulic-diagnostics.com
- **Docs**: [Documentation](../../docs/)

---

## 🎉 Acknowledgments

- PyTorch Team for PyTorch 2.8
- PyTorch Geometric for graph operations
- FastAPI team for excellent async framework
- Python core team for Python 3.13 improvements