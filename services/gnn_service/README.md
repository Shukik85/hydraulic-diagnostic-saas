# GNN Service - Production-Ready ✅

🚀 **UniversalTemporalGNNv2 (GAT + LSTM)** для multi-label classification состояния компонентов гидравлических систем.

## 🎯 Статус: Phase 2 Architecture - v2.1.0 ✅

### ✅ UniversalTemporalGNNv2 v2.1.0 (Phase 2 - Multi-Level Predictions)

**Phase 1 Complete (Декабрь 22, 2025):**
- ✅ **ModelConfig мигрирован** на 6-task архитектуру
- ✅ **6 prediction heads созданы** (4 graph + 2 component)
- ✅ **Lightning Module обновлён** до ModelConfig API
- ✅ **Compatibility tests passing** (6/6 тестов)
- ✅ **Nested output structure** реализована

**Phase 2 Architecture:**

**Graph-level predictions (4 задачи):**
1. **health_score**: `[B, 1]` ∈ [0,1] — Общее здоровье системы (regression)
2. **degradation_rate**: `[B, 1]` ∈ [0,1] — Скорость деградации (regression)
3. **anomaly_flags**: `[B, 9]` ∈ {0,1}^9 — 9 типов аномалий (multi-label)
4. **rul_hours**: `[B, 1]` ∈ [0,∞) — Remaining Useful Life (часы)

**Component-level predictions (2 задачи):**
1. **component_health**: `[N, 1]` ∈ [0,1] — Здоровье каждого компонента
2. **component_anomaly**: `[N, 9]` ∈ {0,1}^9 — Аномалии по компонентам

**Model Stats:**
- ✅ **3,307,799 parameters** (3.3M)
- ✅ **Compatibility check: 6/6 тестов passing**
- ✅ **Single + Temporal modes** поддерживаются
- ✅ **Backward compatible** (алиас UniversalTemporalGNN)

**Осталось (Phase 2):**
- ⚠️ **21 unit test** требуют обновления (старый output format)
- ⚠️ **Integration tests** нужно обновить
- ⚠️ **Inference Engine** проверить совместимость

**Test Suite:**
```bash
# Phase 1 compatibility check (✅ passing)
python scripts/check_phase2_compatibility.py

# Unit tests (⚠️ requires update for new output structure)
pytest tests/test_universal_temporal_gnn.py -v
# Expected: 21 tests need output format updates

# Integration tests
pytest tests/integration/test_full_pipeline.py -v
```

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
└────────────┴─────────────────────────┘
             │
        ┌────────┴────────┐
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
┌────┴───────────┐  ┌────┴─────┐  ┌┴───────────┐
│UniversalTemporal│  │TimescaleDB│  │Prometheus │
│  GNNv2 v2.1.0  │  │ (Sensors) │  │ Metrics   │
│  (Phase 2)     │  │           │  │           │
└────────────────┘  └───────────┘  └───────────┘
```

### 📦 Modular Structure

```
services/gnn_service/
├── src/
│   ├── models/                      # 🆕 UniversalTemporalGNNv2 v2.1.0
│   │   ├── universal_temporal_gnn.py  # ✅ Phase 2 (6 tasks)
│   │   ├── pooling.py                 # AttentionPooling, VirtualNode
│   │   ├── multi_task_loss.py         # ⚠️ Needs update for 6 tasks
│   │   ├── attention_weights.py       # Interpretability
│   │   ├── README.md                  # 📖 Model documentation
│   │   └── __init__.py                # v2.1.0 exports
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   └── validators.py           # Request validation
│   ├── inference/
│   │   ├── inference_engine.py     # ⚠️ Check v2.1.0 compatibility
│   │   ├── model_manager.py        # Model loading
│   │   ├── dynamic_graph_builder.py # Polars-native graph building
│   │   ├── cache.py                # Topology cache
│   │   └── batching.py             # Dynamic batching
│   ├── training/                   # ✅ Phase 2 ready
│   │   ├── dataloader_temporal.py  # Temporal sequences
│   │   ├── lightning_module.py     # ✅ Migrated to ModelConfig API
│   │   ├── losses.py               # Advanced losses
│   │   └── metrics.py              # Multi-level metrics
│   ├── data/                       # Feature engineering
│   ├── schemas/                    # ⚠️ Update PredictionResponse
│   └── middleware/                 # OpenTelemetry, rate limiting
├── tests/
│   ├── test_universal_temporal_gnn.py  # ⚠️ 21 tests need update
│   ├── integration/
│   │   ├── test_full_pipeline.py       # ⚠️ Update for v2.1.0
│   │   └── test_integration_full.py    # ⏸️ Skip (Phase 2)
│   └── unit/                           # 239 unit tests
├── scripts/
│   └── check_phase2_compatibility.py  # ✅ 6/6 passing
├── configs/
│   ├── config.py                   # Configuration
│   └── topology_templates.json     # 📖 Built-in topologies
└── requirements.txt                # Dependencies
```

---

## 🚀 Production Features

### 1. 🧠 UniversalTemporalGNNv2 (v2.1.0)

**Core Architecture:**
```python
from models import UniversalTemporalGNNv2, ModelConfig

# Phase 2 configuration
config = ModelConfig(
    node_features=34,
    edge_features=14,
    gat_hidden_dim=128,
    gat_num_layers=3,
    gat_num_heads=4,
    lstm_hidden_dim=256,
    lstm_num_layers=2,
    # Phase 2: 6 tasks
    graph_anomaly_classes=9,
    component_anomaly_classes=9,
    use_virtual_nodes=True,
    use_attention_pooling=True,
)

model = UniversalTemporalGNNv2(config)

# Forward pass
outputs = model(data, temporal=False)
print(outputs.keys())  # ['component', 'graph']
print(outputs['component'].keys())  # ['health', 'anomaly']
print(outputs['graph'].keys())  # ['health', 'degradation', 'anomaly', 'rul']
```

**Key Features:**
- ✅ **6 prediction tasks**: 4 graph-level + 2 component-level
- ✅ **Nested output structure**: Better organization
- ✅ **Dual mode**: Single graph OR temporal sequences
- ✅ **Size-invariant**: AttentionPooling + VirtualNode
- ✅ **Interpretable**: Attention weights extraction
- ✅ **Production-ready**: Compatibility tested

**Output Format (Phase 2):**
```python
{
    'component': {
        'health': Tensor([N, 1]),      # Per-component health [0,1]
        'anomaly': Tensor([N, 9])      # Per-component anomalies (9 types)
    },
    'graph': {
        'health': Tensor([B, 1]),      # Overall system health [0,1]
        'degradation': Tensor([B, 1]), # Degradation rate [0,1]
        'anomaly': Tensor([B, 9]),     # System anomalies (9 types)
        'rul': Tensor([B, 1])          # Remaining useful life (hours)
    },
    'attention_weights': dict  # Optional, if return_attention=True
}
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
# Example: v2.1.0 (100%) or v2.0.2 (fallback)
config = InferenceConfig(
    model_versions={
        "v2.1.0": ModelConfig(path="v2.1.0.ckpt", traffic=1.0),
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
GET /readyz -> {"ready": true, "model": "v2.1.0", "components": {...}}
```

---

## 🛠️ Configuration

### Environment Variables

```bash
# Service
PORT=8000
HOST=0.0.0.0

# Model
MODEL_VERSION=v2.1.0
MODEL_PATH=models/universal_temporal_gnn_v2.1.0.ckpt
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
gnn_inference_requests_total{model_version="v2.1.0", status}
gnn_inference_errors_total{error_type}

# Latency
gnn_inference_duration_seconds{model_version="v2.1.0"}

# Batching
gnn_inference_batch_size
gnn_request_queue_size

# Model
gnn_model_version{version="2.1.0"}
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
    wget -O models/v2.1.0.ckpt https://your-bucket/v2.1.0.ckpt

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 📝 API Examples

### Diagnose Equipment (Phase 2 Response)

```bash
curl -X POST http://localhost:8000/v1/diagnose \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-123" \
  -d '{
    "equipment_id": "excavator_001",
    "topology_id": "standard_pump_system",
    "timestamp": "2025-12-22T22:00:00Z",
    "sensor_readings": {
      "pump_main": {"pressure": 150.5, "temperature": 65.2},
      "valve_control": {"position": 0.75, "leakage": 0.01}
    }
  }'
```

**Response (Phase 2):**
```json
{
  "status": "success",
  "model_version": "v2.1.0",
  "equipment_id": "excavator_001",
  "timestamp": "2025-12-22T22:00:00Z",
  "diagnosis": {
    "component_predictions": [
      {
        "component_id": "pump_main",
        "health": 0.92,
        "anomalies": {
          "normal": 0.85,
          "pressure_drop": 0.05,
          "overheating": 0.03,
          "cavitation": 0.02,
          "leakage": 0.01,
          "contamination": 0.01,
          "seal_wear": 0.01,
          "bearing_fault": 0.01,
          "valve_stuck": 0.01
        }
      },
      {
        "component_id": "valve_control",
        "health": 0.78,
        "anomalies": {...}
      }
    ],
    "system_predictions": {
      "health": 0.85,
      "degradation_rate": 0.12,
      "anomalies": {
        "normal": 0.75,
        "pressure_drop": 0.15,
        "overheating": 0.05,
        "cavitation": 0.03,
        "leakage": 0.02,
        "contamination": 0.00,
        "seal_wear": 0.00,
        "bearing_fault": 0.00,
        "valve_stuck": 0.00
      },
      "rul_hours": 248.5
    },
    "inference_time_ms": 42.3
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
# Phase 2 compatibility check (✅ passing)
python scripts/check_phase2_compatibility.py

# Model unit tests (⚠️ requires update)
pytest tests/test_universal_temporal_gnn.py -v

# Integration tests
pytest tests/integration/test_full_pipeline.py -v

# All unit tests
pytest tests/unit -v

# Coverage report
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 Documentation

- **[Model Architecture](src/models/README.md)** — UniversalTemporalGNNv2 detailed docs, Phase 2 architecture
- **[Topology Templates](configs/topology_templates.json)** — Built-in hydraulic system templates
- **[API Reference](src/api/README.md)** — FastAPI endpoints (TODO: Update for v2.1.0)
- **[Phase 2 Migration Guide](docs/MIGRATION_PHASE2.md)** — How to update from v2.0.2 to v2.1.0

---

## 📚 References

- **GATv2 Paper**: ["How Attentive are Graph Attention Networks?"](https://arxiv.org/abs/2105.14491) (ICLR 2022)
- **Multi-Level Predictions**: [Issue #116](https://github.com/Shukik85/hydraulic-diagnostic-saas/issues/116)
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

**UniversalTemporalGNNv2 v2.1.0:**
- Phase 2: 6-task multi-level predictions
- 3.3M parameters
- Production-ready architecture
- Backward compatible with v1 alias

**Architecture by:** Senior ML Engineer @ Hydraulic Diagnostics Team

---

## 📄 License

MIT License - see LICENSE file for details
