# GNN Service - Production-Ready ✅

🚀 **UniversalTemporalGNNv2 (GAT + LSTM)** для multi-label classification состояния компонентов гидравлических систем.

## 🎯 Статус: Week 2 Complete - Model Production-Ready!

### ✅ UniversalTemporalGNNv2 v2.0.2 (Production-Hardened)

**Model Status:**
- ✅ **21/21 unit tests passing** (100% success rate)
- ✅ **92% test coverage** для основного модуля
- ✅ **All senior review findings addressed**
- ✅ **Consistent single/temporal mode behavior**
- ✅ **Robust batch handling** (VirtualNodePooling fixed)
- ✅ **Backward compatibility** (v1 alias for migration)

**Architecture:**
- ✅ GATv2 (ICLR 2022) — spatial relationships
- ✅ LSTM — temporal patterns
- ✅ AttentionPooling — size-invariant aggregation
- ✅ VirtualNode — topology-aware representations
- ✅ Multi-task learning — component health + anomaly detection

**Test Suite:**
```bash
# All model tests passing
pytest tests/test_universal_temporal_gnn.py -v
# ✅ 21 passed in 9.23s

# Integration tests updated
pytest tests/integration/test_full_pipeline.py -v
# ✅ 11 tests for training/validation/inference
```

---

## 🏗️ Week 3 Roadmap (Training Pipeline)

### 🎯 Next Steps:
1. **DataLoader для temporal sequences** (TemporalHydraulicDataLoader)
2. **Lightning module integration** (HydraulicGNNModule updates)
3. **Loss functions** (GradientBalanced, Focal, QuantileRUL)
4. **Metrics tracking** (Multi-level metrics)
5. **Checkpoint management** (save/load compatibility)

---

## ⚙️ Requirements

**Python:** 3.14+ (recommended) | 3.11+ (compatible)

- **3.14+** — Full support with PEP 649 (deferred annotations), enhanced asyncio
- **3.11+** — Compatible with legacy fallback for TaskGroup
- **3.10** — Minimum (pipe operator support, but limited features)

**Why Python 3.14?**
- ✅ **PEP 649** — Deferred type annotations (performance boost)
- ✅ **Enhanced asyncio** — Better task management
- ✅ **JIT improvements** — Faster inference
- ✅ **torch.compile()** — JIT compilation for GNN layers

**Dependencies:**
- PyTorch 2.5+ (CUDA 12.9 support)
- PyTorch Geometric 2.7+
- FastAPI 0.115+
- Polars 0.20+ (async-friendly DataFrames)

---

## 🏛️ Architecture

```
┌──────────────────────────────────────┐
│  FastAPI Application (main.py)         │
│  ├─ Request ID Middleware                 │
│  ├─ OpenTelemetry Middleware              │
│  ├─ Rate Limit Middleware (100 req/60s)  │
│  ├─ Body Size Limit (10MB)                │
│  └─ CORS Middleware                       │
└────────────┬─────────────────────────┘
             │
        ┌────────▼────────┐
        │  InferenceEngine  │
        ├──────────────────┤
        │ ● Dynamic Batching│
        │ ● Model Registry  │
        │ ● Topology Cache  │
        │ ● Tensor Validator│
        └────────┬─────────┘
                 │
     ┌───────────┼────────────┐
     │            │            │
┌────▼──────────┐  ┌────▼─────┐  ┌▼───────────┐
│UniversalTemporal│  │TimescaleDB│  │Prometheus │
│  GNNv2 v2.0.2  │  │ (Sensors) │  │ Metrics   │
│  (Production)  │  │           │  │           │
└────────────────┘  └───────────┘  └───────────┘
```

### 📦 Modular Structure

```
services/gnn_service/
├── src/
│   ├── models/                      # 🆕 UniversalTemporalGNNv2
│   │   ├── universal_temporal_gnn.py  # v2.0.2 (Production)
│   │   ├── pooling.py                 # AttentionPooling, VirtualNode
│   │   ├── multi_task_loss.py         # Multi-task learning
│   │   ├── attention_weights.py       # Interpretability
│   │   ├── README.md                  # 📖 Model documentation
│   │   └── __init__.py                # Backward compatibility (v1 alias)
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   └── validators.py           # Request validation
│   ├── inference/
│   │   ├── inference_engine.py     # Main engine (550 lines)
│   │   ├── model_manager.py        # Model loading
│   │   ├── dynamic_graph_builder.py # Polars-native graph building
│   │   ├── cache.py                # Topology cache
│   │   └── batching.py             # Dynamic batching
│   ├── training/                   # 🚧 Week 3 focus
│   │   ├── dataloader_temporal.py  # Temporal sequences
│   │   ├── lightning_module.py     # PyTorch Lightning
│   │   ├── losses.py               # Advanced losses
│   │   └── metrics.py              # Multi-level metrics
│   ├── data/                       # Feature engineering
│   ├── schemas/                    # Pydantic models
│   └── middleware/                 # OpenTelemetry, rate limiting
├── tests/
│   ├── test_universal_temporal_gnn.py  # ✅ 21/21 passing
│   ├── integration/
│   │   ├── test_full_pipeline.py       # ✅ Modernized for v2
│   │   └── test_integration_full.py    # ⏸️ Skip (Week 3)
│   └── unit/                           # 239 unit tests
├── configs/
│   ├── config.py                   # Configuration
│   └── topology_templates.json     # 📖 Built-in topologies
└── requirements.txt                # Dependencies
```

---

## 🚀 Production Features

### 1. 🧠 UniversalTemporalGNNv2 (v2.0.2)

**Core Architecture:**
```python
from models import UniversalTemporalGNNv2, ModelConfig

# Production-ready configuration
config = ModelConfig(
    node_features=34,
    edge_features=14,
    gat_hidden_dim=256,
    gat_num_layers=3,
    gat_num_heads=4,
    lstm_hidden_dim=128,
    lstm_num_layers=2,
    component_health_num_classes=5,
    anomaly_type_num_classes=4,
    use_virtual_nodes=True,
    use_attention_pooling=True,
)

model = UniversalTemporalGNNv2(config)
```

**Key Features:**
- ✅ **Dual mode**: Single graph OR temporal sequences
- ✅ **Size-invariant**: AttentionPooling + VirtualNode
- ✅ **Multi-task**: Component health (node-level) + Anomaly type (graph-level)
- ✅ **Interpretable**: Attention weights extraction
- ✅ **Validated**: 92% test coverage, all edge cases handled

**Production Guarantees:**
```python
# ✅ Consistent predictions across modes
# ✅ Robust batch handling (batch=None gracefully handled)
# ✅ Gradient flow verified (training/validation/test)
# ✅ Deterministic inference (with fixed seed)
# ✅ Device compatibility (CPU/CUDA with fallback)
```

---

### 2. 🔍 OpenTelemetry Distributed Tracing

```python
# Automatic span creation for all requests
# Attributes: method, path, status, duration, errors
# Export to Jaeger, Zipkin, or any OTLP collector
```

**Configuration:**
```bash
OTEL_ENABLED=true
OTEL_SERVICE_NAME=gnn-service
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
```

---

### 3. ⏱️ Rate Limiting (Token Bucket)

```python
# Per-IP rate limiting: 100 requests / 60 seconds
# Headers: X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset
# 429 Too Many Requests response
```

**Configuration:**
```bash
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_S=60
REDIS_URL=redis://localhost:6379  # Optional, falls back to in-memory
```

---

### 4. 📦 Dynamic Batching

```python
# Automatic batching with configurable window
# Max batch size: 32 (configurable)
# Max wait time: 50ms (configurable)
# Race condition protection
```

---

### 5. 🎯 Multi-Model A/B Testing

```python
# Traffic splitting with consistent hashing
# Example: v2.0.2 (100%) or v1 (legacy fallback)
config = InferenceConfig(
    model_versions={
        "v2.0.2": ModelConfig(path="v2.0.2.ckpt", traffic=1.0),
    }
)
```

---

### 6. 💾 Topology Caching + Stampede Protection

```python
# LRU cache with TTL: 100 items, 300s TTL
# AsyncTopologyCache with stampede protection
# Built-in templates: standard_pump_system, dual_pump_system, hydraulic_circuit_type_a
```

**Available Templates:**
- **standard_pump_system** — Single pump (4 components) for excavators, loaders
- **dual_pump_system** — Redundant pumps (7 components) for high-reliability
- **hydraulic_circuit_type_a** — Cooling system (5 components) for industrial

See [configs/topology_templates.json](configs/topology_templates.json) for full documentation.

---

### 7. ⚓ Kubernetes Health Endpoints

```yaml
# Liveness probe
GET /healthz -> {"status": "ok"}

# Readiness probe
GET /readyz -> {"ready": true, "model": "v2.0.2", "components": {...}}
```

---

## 🛠️ Configuration

### Environment Variables

```bash
# Service
PORT=8000
HOST=0.0.0.0

# Model
MODEL_VERSION=v2.0.2
MODEL_PATH=models/universal_temporal_gnn_v2.0.2.ckpt
DEVICE=auto  # cpu, cuda, auto
BATCH_SIZE=32

# Inference
ENABLE_DYNAMIC_BATCHING=true
INFERENCE_TIMEOUT_S=30

# Database
TIMESCALEDB_URL=postgresql://user:pass@localhost:5432/hydraulic
DB_QUERY_TIMEOUT_S=10

# OpenTelemetry
OTEL_ENABLED=true
OTEL_SERVICE_NAME=gnn-service-v2
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_S=60

# Cache
TOPOLOGY_CACHE_SIZE=100
TOPOLOGY_CACHE_TTL_S=300

# Security
MAX_BODY_SIZE_MB=10
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## 📊 Monitoring

### Prometheus Metrics

```
# Requests
gnn_inference_requests_total{model_version="v2.0.2", status}
gnn_inference_errors_total{error_type}

# Latency
gnn_inference_duration_seconds{model_version="v2.0.2"}

# Batching
gnn_inference_batch_size
gnn_request_queue_size

# Model
gnn_model_version{version="2.0.2"}
gnn_model_gpu_memory_bytes{model_version, device}

# Cache
gnn_topology_cache_hit_rate
```

---

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.14-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Download model checkpoint
RUN mkdir -p models && \
    wget -O models/v2.0.2.ckpt https://your-bucket/v2.0.2.ckpt

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📝 API Examples

### Diagnose Equipment (Single Snapshot)

```bash
curl -X POST http://localhost:8000/v1/diagnose \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-123" \
  -d '{
    "equipment_id": "excavator_001",
    "topology_id": "standard_pump_system",
    "timestamp": "2025-12-19T18:00:00Z",
    "sensor_readings": {
      "pump_main": {"pressure": 150.5, "temperature": 65.2},
      "valve_control": {"position": 0.75, "leakage": 0.01}
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "model_version": "v2.0.2",
  "equipment_id": "excavator_001",
  "timestamp": "2025-12-19T18:00:00Z",
  "diagnosis": {
    "component_health": [
      {"component_id": "pump_main", "health_class": "good", "confidence": 0.92},
      {"component_id": "valve_control", "health_class": "warning", "confidence": 0.78}
    ],
    "anomaly_type": {
      "predictions": {
        "normal": 0.75,
        "pressure_drop": 0.15,
        "overheating": 0.05,
        "cavitation": 0.03,
        "leakage": 0.02
      }
    },
    "inference_time_ms": 42.3,
    "attention_weights": {...}  // Optional, for interpretability
  }
}
```

---

## 👥 Development

### Local Setup

```bash
# Clone repo
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas/services/gnn_service

# Create venv (Python 3.14 recommended)
python3.14 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy env template
cp .env.example .env

# Run service
uvicorn src.api.main:app --reload
```

### Testing

```bash
# Model unit tests (21 tests, 92% coverage)
pytest tests/test_universal_temporal_gnn.py -v --cov=src/models

# Integration tests
pytest tests/integration/test_full_pipeline.py -v

# All unit tests
pytest tests/unit -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 Documentation

- **[Model Architecture](src/models/README.md)** — UniversalTemporalGNNv2 detailed docs, configuration, migration guide
- **[Topology Templates](configs/topology_templates.json)** — Built-in hydraulic system templates
- **[API Reference](src/api/README.md)** — FastAPI endpoints (TODO: Week 3)

---

## 📚 References

- **GATv2 Paper**: ["How Attentive are Graph Attention Networks?"](https://arxiv.org/abs/2105.14491) (ICLR 2022)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenTelemetry](https://opentelemetry.io/)
- [Prometheus](https://prometheus.io/)

---

## 🎉 Credits

**Built with:**
- Python 3.14 (PEP 649, enhanced asyncio)
- PyTorch 2.5+ & PyTorch Geometric 2.7+
- FastAPI 0.115+
- Polars 0.20+
- OpenTelemetry 1.28+

**UniversalTemporalGNNv2 v2.0.2:**
- Senior review findings addressed
- Production-hardened architecture
- 21/21 tests passing (92% coverage)

**Architecture by:** Senior ML Engineer @ Hydraulic Diagnostics Team

---

## 📄 License

MIT License - see LICENSE file for details
