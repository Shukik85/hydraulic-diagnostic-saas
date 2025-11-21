# GNN Service - Production-Ready Implementation

🌱 **Status:** In Active Development  
🔗 **Branch:** `feature/gnn-service-production-ready`  
📅 **Created:** 2025-11-21  
🎯 **Epic Issue:** [#92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)

---

## 🚀 Overview

Production-ready Graph Neural Network service для диагностики гидравлических систем с использованием **Universal Temporal GNN** (GAT + LSTM).

### Technology Stack (Updated 2025-11-21)

- 🐍 **Python 3.14.0** - Free-threaded mode, deferred annotations, t-strings
- ⚡ **PyTorch 2.8.0** - Float8 training, quantized inference, torch.compile
- 🖥️ **CUDA 12.9** - Family-specific optimizations, Blackwell support
- 🧠 **PyTorch Lightning 2.1+** - Structured training pipeline
- 🚀 **FastAPI 0.109+** - Async API framework
- ✅ **Pydantic v2.6+** - Data validation with deferred annotations
- 📊 **TimescaleDB** - Time-series sensor data
- 🔄 **Redis** - Caching layer

### Key Features

- ✅ **Clean Architecture** - Modular `src/` organization, zero stub files
- 🧠 **Universal Temporal GNN** - GAT (Graph Attention) + LSTM for time-series
- ⚡ **Modern Python** - Free-threading (no GIL), deferred annotations
- 🔥 **PyTorch 2.8** - Float8 training, torch.compile, quantized inference
- 🚀 **FastAPI** - Async/await, Pydantic v2 validation
- 📊 **Observability** - Structured logging, Prometheus metrics
- 🔄 **Production Pipeline** - PyTorch Lightning, distributed training (DDP)
- 🐳 **Containerized** - Docker with CUDA 12.9 support

---

## 📋 Current Status

### ✅ Phase 1 - Week 1 (Foundation)

**Completed (2025-11-21):**
- [x] Repository structure cleanup
- [x] Legacy files archived to `_legacy/`
- [x] New `src/` modular structure
- [x] Documentation written
- [x] Dependencies updated (Python 3.14, PyTorch 2.8, CUDA 12.9)
- [x] Epic Issue #92 created
- [x] Sub-Issues #93-96 created

**In Progress:**
- [ ] [#93](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/93) - Core Schemas Implementation (8h)
- [ ] [#94](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/94) - GNN Model Architecture (12h)
- [ ] [#95](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/95) - Dataset & DataLoader (14h)
- [ ] [#96](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/96) - Inference Engine (10h)

### 🔲 Phase 2 - Week 2 (Training & Integration)
- Training pipeline (PyTorch Lightning)
- Distributed training (DDP)
- Float8 training integration
- FastAPI ↔ TimescaleDB
- Model management

### 🔲 Phase 3 - Week 3 (Production Hardening)
- Observability (logging, metrics)
- Error handling & resilience
- Comprehensive testing
- API documentation
- Deployment (Docker, K8s)

---

## 📚 Quick Start

### Prerequisites

- **Python 3.14.0+** (required for free-threading)
- **CUDA 12.9+** (for GPU support)
- **Docker 24+** (optional)
- **TimescaleDB 2.14+** (for sensor data)

### Installation

```bash
# Clone repository
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas

# Checkout feature branch
git checkout feature/gnn-service-production-ready

# Navigate to service
cd services/gnn_service

# Create virtual environment (Python 3.14)
python3.14 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Configure environment
cp .env.example .env
# Edit .env with your configuration

# Run service
uvicorn api.main:app --reload --port 8002
```

### Docker Development

```bash
# Build dev image (Python 3.14 + CUDA 12.9)
docker build -f Dockerfile.dev -t gnn-service:dev .

# Run with hot reload
docker run -p 8002:8002 \
  -v $(pwd):/app \
  --gpus all \
  --env-file .env \
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
  -e REDIS_URL=redis://... \
  gnn-service:latest
```

---

## 🏛️ Architecture

### Directory Structure

```
services/gnn_service/
├── src/                      # Source code (clean implementation)
│   ├── models/              # GNN models
│   │   ├── gnn_model.py     # UniversalTemporalGNN (GAT + LSTM)
│   │   ├── layers.py        # Custom layers
│   │   └── attention.py     # Attention mechanisms
│   ├── data/                # Data processing
│   │   ├── dataset.py       # HydraulicGraphDataset
│   │   ├── loader.py        # DataLoader factory
│   │   ├── preprocessing.py # Feature engineering
│   │   └── graph_builder.py # Graph construction
│   ├── inference/           # Inference engine
│   │   ├── engine.py        # InferenceEngine
│   │   ├── post_processing.py
│   │   └── batch_processor.py
│   ├── training/            # Training pipeline
│   │   ├── trainer.py       # GNNTrainer (Lightning)
│   │   ├── callbacks.py     # Training callbacks
│   │   └── metrics.py       # Custom metrics
│   ├── schemas/             # Pydantic models
│   │   ├── graph.py         # Graph schemas
│   │   ├── metadata.py      # Metadata schemas
│   │   ├── requests.py      # API requests
│   │   └── responses.py     # API responses
│   └── utils/               # Utilities
│       ├── device.py        # CUDA management
│       └── checkpointing.py # Model checkpoints
├── api/                     # FastAPI application
│   ├── main.py
│   ├── routes/
│   │   ├── inference.py
│   │   ├── admin.py
│   │   ├── monitoring.py
│   │   └── health.py
│   └── middleware/
├── config/                  # Configuration
│   ├── settings.py
│   └── database.py
├── tests/                   # Tests
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── _legacy/                 # Archived code
├── data/                    # Data directory
├── models/                  # Saved models
├── logs/                    # Logs
└── docs/                    # Documentation
```

See [STRUCTURE.md](STRUCTURE.md) for detailed architecture.

### GNN Model Architecture

```
Sensor Time-Series Data
        ↓
   Preprocessing
   (Feature Engineering)
        ↓
   Graph Construction
   (Dynamic Topology)
        ↓
   ╔═══════════════════╗
   ║ UniversalTemporalGNN ║
   ╚═══════════════════╝
        ↓
   GAT Layers (×3)
   - Multi-head attention
   - Spatial relationships
   - Layer normalization
        ↓
   LSTM Layers (×2)
   - Temporal modeling
   - Sequence learning
        ↓
   Output Heads:
   ├─ Health Score (0-1)
   ├─ Degradation Rate
   └─ Anomaly Detection (3 types)
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
  "version": "2.0.0",
  "status": "healthy",
  "timestamp": "2025-11-21T01:00:00Z",
  "checks": {
    "model": "loaded",
    "database": "connected",
    "gpu": "available",
    "redis": "connected"
  },
  "stack": {
    "python": "3.14.0",
    "pytorch": "2.8.0",
    "cuda": "12.9"
  }
}
```

### Inference Request

```bash
curl -X POST http://localhost:8002/api/v1/inference \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "equipment_id": "excavator_001",
    "time_window": {
      "start_time": "2025-11-01T00:00:00Z",
      "end_time": "2025-11-13T00:00:00Z"
    },
    "include_recommendations": true,
    "return_attention": false
  }'
```

**Response:**
```json
{
  "request_id": "req_1732147500000",
  "equipment_id": "excavator_001",
  "overall_health_score": 0.87,
  "component_health": [
    {
      "component_id": "pump_main",
      "component_type": "hydraulic_pump",
      "health_score": 0.92,
      "degradation_rate": 0.02,
      "confidence": 0.95,
      "status": "healthy"
    },
    {
      "component_id": "valve_01",
      "component_type": "hydraulic_valve",
      "health_score": 0.78,
      "degradation_rate": 0.08,
      "confidence": 0.89,
      "status": "warning"
    }
  ],
  "anomalies": [
    {
      "anomaly_id": "anom_001",
      "anomaly_type": "pressure_drop",
      "severity": "medium",
      "confidence": 0.85,
      "affected_components": ["valve_01"],
      "description": "Unusual pressure fluctuation detected in valve_01",
      "detected_at": "2025-11-12T14:23:00Z"
    }
  ],
  "recommendations": [
    "Schedule maintenance for valve_01 within 7 days",
    "Monitor pump_main pressure levels daily",
    "Check hydraulic fluid quality"
  ],
  "metadata": {
    "inference_time_ms": 387.5,
    "model_version": "2.0.0",
    "timestamp": "2025-11-21T01:00:00Z",
    "device": "cuda:0"
  }
}
```

### Batch Inference

```bash
curl -X POST http://localhost:8002/api/v1/batch-inference \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"equipment_id": "exc_001", "time_window": {...}},
      {"equipment_id": "exc_002", "time_window": {...}}
    ]
  }'
```

### API Documentation

**Interactive docs:**
- **Swagger UI:** http://localhost:8002/docs
- **ReDoc:** http://localhost:8002/redoc
- **OpenAPI JSON:** http://localhost:8002/openapi.json

---

## 🧪 Testing

### Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html --cov-report=term

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Specific test file
pytest tests/unit/test_models.py

# Specific test
pytest tests/unit/test_models.py::test_gnn_forward_pass

# Parallel testing (pytest-xdist)
pytest -n auto
```

### Code Quality

```bash
# Format code
black src/ api/ tests/
isort src/ api/ tests/

# Lint
ruff check src/ api/ tests/

# Type check (Python 3.14 strict mode)
mypy src/ api/ --strict

# Security scan
bandit -r src/ api/

# All checks
./scripts/code_quality.sh
```

---

## 📊 Monitoring

### Prometheus Metrics

```bash
curl http://localhost:8002/metrics
```

**Key Metrics:**
```
# Inference latency
gnn_inference_duration_seconds{quantile="0.5"} 0.187
gnn_inference_duration_seconds{quantile="0.95"} 0.453
gnn_inference_duration_seconds{quantile="0.99"} 0.687

# Request counters
gnn_inference_total{status="success"} 1547
gnn_inference_errors_total{error_type="timeout"} 3

# GPU metrics
gnn_gpu_utilization_percent 78.5
gnn_gpu_memory_allocated_mb 2048.3

# Model metrics
gnn_model_load_time_seconds 2.34
gnn_batch_size_avg 24.7
```

### Health Endpoints

```bash
# Liveness probe (is service running?)
curl http://localhost:8002/health/live

# Readiness probe (ready to serve traffic?)
curl http://localhost:8002/health/ready

# Detailed health
curl http://localhost:8002/health
```

---

## 🎓 Training

### Prepare Dataset

```bash
# Preprocess raw sensor data
python -m src.data.preprocessing \
  --input data/raw/sensor_data.csv \
  --output data/processed/ \
  --metadata data/metadata/equipment.json \
  --window-size 60 \
  --sequence-length 10
```

### Train Model

```bash
# Single GPU training
python -m src.training.train \
  --config config/training.yaml \
  --data data/processed/ \
  --output models/checkpoints/ \
  --gpus 1

# Multi-GPU (DDP)
python -m src.training.train \
  --config config/training.yaml \
  --data data/processed/ \
  --output models/checkpoints/ \
  --gpus 4 \
  --strategy ddp

# Float8 training (PyTorch 2.8 - 1.5x faster)
python -m src.training.train \
  --config config/training.yaml \
  --data data/processed/ \
  --float8-training \
  --gpus 4
```

### Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# Weights & Biases (optional)
wandb login
python -m src.training.train --wandb --project hydraulic-gnn
```

---

## 🐳 Deployment

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f gnn-service

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Apply manifests
kubectl apply -f kubernetes/

# Check deployment
kubectl get pods -n hydraulic-prod
kubectl get svc -n hydraulic-prod

# View logs
kubectl logs -f deployment/gnn-service -n hydraulic-prod

# Port forward for testing
kubectl port-forward svc/gnn-service 8002:8002 -n hydraulic-prod
```

---

## 📖 Documentation

### Project Documentation
- **[Epic Issue #92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)** - Main tracking issue
- **[Roadmap](../../docs/GNN_SERVICE_ROADMAP.md)** - 3-week implementation plan
- **[Structure](STRUCTURE.md)** - Architecture details
- **[Migration Summary](MIGRATION_SUMMARY.md)** - Migration documentation
- **[Legacy README](_legacy/README_LEGACY.md)** - Old code archive

### API Documentation
- **Swagger UI:** http://localhost:8002/docs
- **ReDoc:** http://localhost:8002/redoc

### Technology Documentation
- [Python 3.14 What's New](https://docs.python.org/3.14/whatsnew/3.14.html)
- [PyTorch 2.8 Release](https://dev-discuss.pytorch.org/t/pytorch-release-2-8-key-information/3039)
- [CUDA 12.9 Features](https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## 🤝 Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Workflow

1. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes following code style**

3. **Write tests (coverage ≥ 80%)**

4. **Run code quality checks:**
   ```bash
   black src/ tests/
   ruff check src/ tests/
   mypy src/ --strict
   pytest --cov=src
   ```

5. **Commit with conventional commits:**
   ```bash
   git commit -m "feat(inference): add batch optimization"
   ```

6. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```

---

## ⚠️ Known Issues

**Current limitations:**
- [ ] Python 3.14 - некоторые библиотеки могут иметь проблемы совместимости
- [ ] PyTorch 2.8 - float8 training требует A100/H100 GPU
- [ ] CUDA 12.9 - Maxwell/Pascal/Volta deprecated (только support)

**Workarounds:**
- Используйте CPU inference для разработки
- Float8 опционален (по умолчанию отключен)
- Старые GPU поддерживаются в compatibility mode

---

## 🔗 Links

### GitHub
- **Repository:** [Shukik85/hydraulic-diagnostic-saas](https://github.com/Shukik85/hydraulic-diagnostic-saas)
- **Branch:** [feature/gnn-service-production-ready](https://github.com/Shukik85/hydraulic-diagnostic-saas/tree/feature/gnn-service-production-ready)
- **Issues:** [Project Issues](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues)
- **Epic:** [Issue #92](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/92)

### Phase 1 Issues
- [#93 - Core Schemas](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/93)
- [#94 - GNN Model](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/94)
- [#95 - Dataset & DataLoader](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/95)
- [#96 - Inference Engine](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/96)

---

## 📧 Support

- **GitHub Issues:** [Create Issue](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/new)
- **Email:** shukik85@ya.ru
- **Documentation:** [docs/](../../docs/)

---

## 🏆 Acknowledgments

- **Python Team** for Python 3.14 with free-threading
- **PyTorch Team** for PyTorch 2.8 and float8 training
- **NVIDIA** for CUDA 12.9 and Blackwell support
- **PyTorch Geometric** for graph neural network operations
- **FastAPI Team** for excellent async framework
- **PyTorch Lightning** for structured training

---

**Last Updated:** 2025-11-21 04:00 MSK  
**Status:** 🚧 Active Development  
**Next Milestone:** Phase 1 Complete (Nov 27, 2025)