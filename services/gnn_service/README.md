# GNN Service - Universal Temporal GNNv2 Phase 2

🚀 **UniversalTemporalGNNv2 (GAT + LSTM)** для multi-label classification состояния компонентов гидравлических систем.

## ⚠️ PHASE 2 INTEGRATION CHECKLIST

### Model & Code Status

- ✅ **UniversalTemporalGNNv2 v2.1.0** — 25KB production code, 6-task architecture
- ✅ **Unit Tests** — 21 tests written for v2.1.0 output structure
- ✅ **FastAPI Application** — Middleware, K8s probes, error handling complete
- ❌ **Inference Engine** — Expects tuple, model returns dict (INCOMPATIBLE)
- ❌ **Response Schema** — Missing component-level fields and RUL
- ❌ **Integration Testing** — Cannot run (blocked by inference incompatibility)

### Blocking Issue

**File**: `src/inference/inference_engine.py` line ~320  
**Problem**: Output format mismatch

```python
# Model returns:
outputs = {
    'component': {...},
    'graph': {...}
}

# Inference tries:
health, degradation, anomaly = model(...)  # CRASHES
```

**Status**: 🔴 DO NOT DEPLOY  
**Fix Required**: 4-6 hours  
**Blocking**: Yes

---

## ✅ Complete Components

### Model: UniversalTemporalGNNv2 v2.1.0

**Output Structure (6 Tasks)**:
```python
{
    'component': {
        'health': Tensor([N, 1]),          # Per-component health [0,1]
        'anomaly': Tensor([N, 9])          # Per-component anomalies (9 types)
    },
    'graph': {
        'health': Tensor([B, 1]),          # Overall system health [0,1]
        'degradation': Tensor([B, 1]),     # Degradation rate [0,1]
        'anomaly': Tensor([B, 9]),         # System anomalies (9 types)
        'rul': Tensor([B, 1])              # Remaining useful life (hours)
    },
    'attention_weights': dict              # Optional, if return_attention=True
}
```

**Features**:
- ✅ 6 prediction tasks (4 graph + 2 component)
- ✅ Nested dict structure
- ✅ Dual mode (single graph + temporal sequences)
- ✅ Size-invariant (AttentionPooling + VirtualNode)
- ✅ 3.3M parameters
- ✅ Production-grade code quality

**Statistics**:
- Parameters: 3,307,799
- GAT layers: Configurable (2-4)
- LSTM layers: Configurable (1-2)
- Code: 700+ lines, zero TODOs

### Unit Tests

**File**: `tests/test_universal_temporal_gnn.py`

- ✅ 21 tests written for v2.1.0
- ✅ Tests verify nested dict output
- ✅ Tests check all 6 outputs
- ✅ Coverage: config, forward, attention, temporal, batch, gradients
- ❌ Cannot run (blocked by inference incompatibility)

### FastAPI Application

**File**: `src/api/main.py`

- ✅ OpenTelemetry middleware (distributed tracing)
- ✅ Request ID tracking
- ✅ Rate limiting (100 req/60s per IP)
- ✅ CORS middleware
- ✅ Body size limit (10MB)
- ✅ K8s health endpoints (/healthz, /readyz)
- ✅ Error handling with proper HTTP codes
- ✅ Async/await throughout

---

## ❌ Components Requiring Fixes

### Fix 1: Inference Engine (2-3 hours)

**File**: `src/inference/inference_engine.py`

**Current Code (BROKEN)**:
```python
def _inference_single(self, graph: Data, model_version: str) -> tuple:
    with torch.inference_mode():
        health, degradation, anomaly = model(...)  # UNPACKS DICT → CRASHES
    return health, degradation, anomaly
```

**Required Changes**:
1. Change to `-> dict` return type
2. Remove tuple unpacking: `outputs = model(...)`
3. Return full dict: `return outputs`
4. Update `_postprocess()` to handle dict structure
5. Extract component and graph outputs separately
6. Add component-level predictions to response
7. Add RUL to response

### Fix 2: Response Schema (1 hour)

**File**: `src/schemas/responses.py`

**Required Changes**:
1. Add `rul_hours: float` field
2. Add `component_predictions: List[ComponentDiagnosis]`
3. Update docstrings for new fields

### Fix 3: Tests (0.5 hours)

**Run**:
```bash
pytest tests/test_universal_temporal_gnn.py -v
# Should pass after inference_engine fixes
```

### Fix 4: Integration Tests (1-2 hours)

**Run**:
```bash
pytest tests/integration/test_full_pipeline.py -v
# Should pass after all fixes
```

---

## 🏛️ Architecture

```
┌────────────────────────────────┐
│  FastAPI Application  ✅ GOOD  │
│  ├─ Request ID                 │
│  ├─ OpenTelemetry              │
│  ├─ Rate Limit                 │
│  └─ CORS                        │
└────────────┬────────────────────┘
             │
        ┌────▼─────────────────────┐
        │ InferenceEngine ❌ BROKEN  │
        │ ├─ Dynamic Batching       │
        │ ├─ Model Registry         │
        │ ├─ Topology Cache         │
        │ └─ Tensor Validator       │
        └────┬────────────────────┘
             │
  ┌──────────┼──────────┐
  │          │          │
  ▼          ▼          ▼
Model      Database   Monitoring
✅ GOOD    ✅ GOOD    ✅ GOOD
```

---

## 📦 Directory Structure

```
services/gnn_service/
├── src/
│   ├── models/                           # ✅ COMPLETE
│   │   ├── universal_temporal_gnn.py     # v2.1.0 (6 tasks)
│   │   ├── pooling.py                    # AttentionPooling
│   │   └── __init__.py
│   ├── inference/                        # ❌ NEEDS FIX
│   │   ├── inference_engine.py           # Fix: tuple → dict
│   │   ├── model_manager.py              # ✅
│   │   └── cache.py                      # ✅
│   ├── api/                              # ✅ GOOD
│   │   ├── main.py                       # FastAPI app
│   │   └── validators.py
│   ├── schemas/                          # ❌ NEEDS FIELDS
│   │   └── responses.py                  # Add RUL + components
│   └── training/                         # ✅
│       ├── lightning_module.py
│       └── metrics.py
├── tests/
│   ├── test_universal_temporal_gnn.py    # ✅ 21 tests
│   └── integration/                      # ❌ Can't run yet
├── configs/
│   ├── config.py
│   └── topology_templates.json
└── requirements.txt
```

---

## 🚀 Development

### Setup

```bash
cd services/gnn_service
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Run Service

```bash
uvicorn src.api.main:app --reload
```

### Run Tests

```bash
# Model tests (will fail due to inference issue)
pytest tests/test_universal_temporal_gnn.py -v

# Integration tests (will fail due to inference issue)
pytest tests/integration/ -v
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Service
PORT=8000
HOST=0.0.0.0

# Model
MODEL_VERSION=v2.1.0
MODEL_PATH=models/universal_temporal_gnn_v2.1.0.ckpt
DEVICE=auto

# Inference
BATCH_SIZE=32
ENABLE_DYNAMIC_BATCHING=true
INFERENCE_TIMEOUT_S=30

# Database
TIMESCALEDB_URL=postgresql://user:pass@localhost:5432/hydraulic

# OpenTelemetry
OTEL_ENABLED=true
OTEL_SERVICE_NAME=gnn-service-v2
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_S=60
```

---

## 📊 Monitoring

### Health Endpoints

```bash
# Liveness
curl http://localhost:8000/healthz
# {"status": "ok"}

# Readiness
curl http://localhost:8000/readyz
# {"ready": true, "model": "v2.1.0", ...}
```

### Prometheus Metrics

```
gnn_inference_requests_total
gnn_inference_duration_seconds
gnn_inference_errors_total
gnn_inference_batch_size
gnn_topology_cache_hit_rate
```

---

## 🎯 Deployment Status

| Aspect | Status | Note |
|--------|--------|------|
| Model Code | ✅ | Production-ready, 6 tasks implemented |
| Unit Tests | ✅ | 21 tests written, correct assertions |
| API Server | ✅ | FastAPI with proper middleware |
| Inference | ❌ | Output format mismatch (tuple vs dict) |
| Response Schema | ❌ | Missing component-level fields |
| Integration | ❌ | Cannot test due to inference issue |
| **Can Deploy?** | **🔴 NO** | Must fix inference first |

**Deployment Timeline**:
- Fix inference engine: 2-3 hours
- Fix schemas: 1 hour
- Test: 1-2 hours
- **Total**: 4-6 hours until ready

---

## 📚 Documentation Files

- **[Model Code](src/models/universal_temporal_gnn.py)** — 6-task architecture
- **[Unit Tests](tests/test_universal_temporal_gnn.py)** — 21 tests for v2.1.0
- **[Configuration](configs/topology_templates.json)** — Built-in topologies
- **[Topology Reference](configs/topology_templates.json)** — System templates

---

## 🔗 Dependencies

- Python 3.14+
- PyTorch 2.5+
- PyTorch Geometric 2.7+
- FastAPI 0.115+
- Polars 0.20+
- OpenTelemetry 1.28+

---

## 📝 Next Steps

1. Read this README (Phase 2 status)
2. Fix inference_engine.py (2-3 hours)
3. Update response schemas (1 hour)
4. Run tests (should pass)
5. Run integration tests (1-2 hours)
6. Deploy (when ready)

---

**Last Updated**: January 3, 2026  
**Status**: Phase 2 Model Complete, Integration In Progress  
**Deployment**: 🔴 NOT READY - Fix inference first
