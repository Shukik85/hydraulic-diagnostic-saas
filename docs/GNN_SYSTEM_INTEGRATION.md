# GNN System Integration Guide

## Обзор

GNN Service — это **"мозг системной диагностики"**, который видит гидравлическую систему как **граф взаимосвязанных компонентов**.

### Ключевые особенности

- **System-level analysis**: Понимает каскадные эффекты (pump → boom → stick)
- **Explainable AI**: Attention механизм показывает **почему** обнаружена аномалия
- **Real-time capable**: <50ms inference latency (p90)
- **Fleet management**: Batch inference для парка техники

---

## Архитектура

```
ПОЛЬЗОВАТЕЛЬ (Оператор/Инженер/Диспетчер)
         │
         │ UI/UX
         ↓
┌────────────────────────────────────────┐
│   FRONTEND (Nuxt 4 + Digital Twin)    │
│  - System Health Dashboard            │
│  - Interactive Graph View             │
│  - Real-time Alerts + Reasoning       │
└────────────────────────────────────────┘
         │
         │ WebSocket + REST
         ↓
┌────────────────────────────────────────┐
│   BACKEND (Django + DRF)              │
│                                        │
│  Orchestrator Layer:                  │
│  - DiagnosticCoordinator              │
│  - GraphBuilder                       │
│  - WebSocket Manager                  │
│  - Result Aggregator                  │
└────────────────────────────────────────┘
         │
    ┌────┼────┐
    │         │
    ↓         ↓
┌────────┐  ┌────────────┐  ┌──────────┐
│  GNN   │  │ Component  │  │   RAG    │
│ Service│  │   Models   │  │ Assistant│
│        │  │ (ml_service)│  │ (Qwen3)  │
│ T-GAT  │  │            │  │          │
│ Attn   │  │ - Pump     │  │ - Docs   │
│ Explain│  │ - Cylinder │  │ - History│
└────────┘  └────────────┘  └──────────┘
    │            │            │
    └────────┼────────────┘
              │
         ┌────┼────┐
         │         │
    ┌────┴─────────┼───┐
    │                     │
┌────────────┐      ┌───────────┐
│ TimescaleDB│      │   Redis   │
│ - Sensors  │      │ - Cache   │
│ - Topology │      │ - Celery  │
│ - Results  │      └───────────┘
└────────────┘
```

---

## Сценарий A: Real-time мониторинг

### Пользователь: Оператор экскаватора

#### Шаг 1: Датчики → Backend

```python
# Каждые 100ms: давление, температура, обороты, положение
Modbus Gateway → Backend Ingestion API
SensorData.objects.bulk_create([
    SensorData(equipment_id=1, sensor_type="pressure", value=185.0),
    SensorData(equipment_id=1, sensor_type="temperature", value=68.0),
    ...
])
```

#### Шаг 2: Backend Orchestrator

```python
from diagnostics.coordinator import DiagnosticCoordinator

coordinator = DiagnosticCoordinator()
result = await coordinator.run_diagnostics(
    equipment_id=1,
    mode="hybrid",  # GNN + component models
)
```

#### Шаг 3: GNN Service Inference

**Request**:
```json
{
  "node_features": [
    [185.0, 68.0, 180.0, 2.1, ...],  // pump
    [165.0, 72.0, 120.0, 1.8, ...],  // boom
    [160.0, 70.0, 110.0, 1.5, ...],  // stick
    ...
  ],
  "edge_index": [[0, 0, 1], [1, 2, 2]],  // pump→boom, pump→stick, boom→stick
  "component_names": ["pump", "boom", "stick", "bucket"]
}
```

**Response (<50ms)**:
```json
{
  "prediction": 1,  // ANOMALY
  "probability": 0.94,
  "anomaly_score": 0.89,
  "explanation": {
    "critical_components": ["pump", "boom"],
    "attention_scores": [0.82, 0.67, 0.15, 0.08],
    "causal_path": [
      "pump_critical_failure",
      "boom_degradation"
    ],
    "reasoning": "Primary component affected: pump. Secondary components: boom. Causal chain detected: pump_critical_failure → boom_degradation. System-level anomaly detected with cascading effects. Immediate inspection recommended."
  }
}
```

#### Шаг 4: Component-level Analysis

Если GNN обнаружил аномалию, Backend вызывает ml_service для critical компонентов:

```python
# ml_service: pump model
result = await ml_client.predict(equipment_id=1, model_type="pump")
# {
#   "prediction": "bearing_wear",
#   "confidence": 0.95,
#   "diagnosis": "Fe particles detected in oil sample",
#   "recommended_action": "Replace bearing within 24h"
# }
```

#### Шаг 5: Aggregation + WebSocket Push

```python
# Backend sends to frontend via WebSocket
ws_manager.broadcast(
    channel=f"equipment/{equipment_id}",
    message={
        "type": "diagnostic_alert",
        "data": {
            "system": gnn_result,
            "components": component_results,
            "recommendation": recommendation,
        }
    }
)
```

#### Шаг 6: Frontend UI

```
┌────────────────────────────────────────┐
│ 🔴 SYSTEM ANOMALY (GNN)                      │
│ Confidence: 94% | Score: 0.89                │
├────────────────────────────────────────┤
│ 🔍 System Analysis:                       │
│ Affected: Pump → Boom Cylinder              │
│                                            │
│ Causal Chain:                              │
│ 🔥 Pump overheating → 📉 Pressure drop       │
│                                            │
│ 🔬 Component Details:                     │
│                                            │
│ 💧 Pump: Bearing wear (95% conf)         │
│   Action: Replace bearing within 24h       │
│                                            │
│ 🔧 Boom: Seal degradation (89% conf)      │
│   Action: Inspect seals, plan repair       │
├────────────────────────────────────────┤
│ ⚠️  PRIORITY: CRITICAL                      │
│ 🕒 Timeframe: Immediate                    │
│                                            │
│ [View Graph] [Ask AI] [Acknowledge]       │
└────────────────────────────────────────┘
```

---

## API Endpoints

### GNN Service

#### POST /predict

Single equipment inference.

**Request**:
```json
{
  "node_features": [[...]],
  "edge_index": [[...]],
  "edge_attr": [[...]] (optional),
  "component_names": ["pump", "boom", ...]
}
```

**Response**:
```json
{
  "prediction": 0 | 1,
  "probability": 0.95,
  "anomaly_score": 0.89,
  "explanation": {...} (if anomaly)
}
```

#### POST /batch_predict

Fleet batch inference.

**Request**:
```json
{
  "graphs": [
    {"equipment_id": "CAT-336-001", "node_features": [...], ...},
    {"equipment_id": "CAT-336-002", ...},
    ...
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {"equipment_id": "CAT-336-001", "prediction": 1, "anomaly_score": 0.89},
    ...
  ]
}
```

#### GET /health

Health check.

#### GET /metrics

Prometheus metrics.

---

## Local Development

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU + nvidia-docker2 (для GNN + Ollama)
- Python 3.11+
- Node.js 20+ (для frontend)

### Quick Start

```bash
# 1. Clone repo
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas
git checkout feature/gnn-system-integration

# 2. Setup environment
cp .env.example .env
# Edit .env: set GNN_INTERNAL_API_KEY, ML_INTERNAL_API_KEY, RAG_INTERNAL_API_KEY

# 3. Build base GPU image
docker build -f docker/base-ai-gpu.Dockerfile -t hydraulic-ai-base-gpu:cuda12.1 .

# 4. Start services
docker-compose up -d db redis
docker-compose up -d gnn_service ml_service rag_service ollama
docker-compose up -d backend celery beat
docker-compose up -d frontend

# 5. Check health
curl http://localhost:8003/health  # GNN Service
curl http://localhost:8001/health  # ML Service
curl http://localhost:8002/health  # RAG Service
curl http://localhost:8000/health/ # Backend

# 6. Test inference
curl -X POST http://localhost:8003/predict \
  -H "X-Internal-API-Key: your-gnn-key" \
  -H "Content-Type: application/json" \
  -d '{
    "node_features": [[185.0, 68.0, 180.0, 2.1, 0, 0, 0, 0, 0, 0]],
    "edge_index": [[0], [0]],
    "component_names": ["pump"]
  }'
```

### GPU Memory Management

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Expected allocation:
# - GNN Service: ~2-3 GB
# - Ollama (DeepSeek-R1): ~8-10 GB
# Total: ~10-13 GB (требуется GPU с ≥16 GB VRAM)
```

---

## Production Deployment

### Kubernetes

See `k8s/gnn-service-deployment.yaml` for full configuration.

**Key points**:

- GPU node pools with NVIDIA drivers
- Horizontal Pod Autoscaler (3-10 replicas)
- PersistentVolumeClaim for model storage
- Service mesh (Istio) for traffic management
- Prometheus + Grafana for monitoring

---

## Performance Benchmarks

| Metric | Target | Status |
|--------|--------|--------|
| GNN Inference (p90) | <100ms | ⏳ Pending GPU testing |
| GNN Inference (p50) | <50ms | ⏳ Pending GPU testing |
| Batch Inference (50 graphs) | <2s | ⏳ Pending testing |
| Attention Explainability | <10ms | ✅ Implemented |
| Model Load Time | <30s | ✅ Implemented |

---

## Troubleshooting

### GNN Service won't start

**Symptom**: `Model not loaded` error

**Solution**:
1. Check model path: `MODEL_PATH=/models/gnn_classifier_best.ckpt`
2. For dev: GNN Service uses random weights if model not found
3. Check logs: `docker logs hdx-gnn-service`

### GPU not detected

**Symptom**: `CUDA not available`

**Solution**:
```bash
# Check nvidia-docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi

# Verify docker-compose GPU config
docker-compose config | grep -A5 devices
```

### High latency (>100ms)

**Symptom**: Slow inference

**Causes**:
1. Model on CPU instead of GPU
2. Large graph (>100 nodes)
3. Network overhead (Backend→GNN)

**Solution**:
- Set `DEVICE=cuda` in .env
- Enable GPU in docker-compose
- Use batch inference for fleet

---

## Next Steps

1. **Training**: Self-Supervised GNN pretraining on UCI dataset
2. **Frontend**: Digital Twin 3D viewer + Graph View
3. **RAG Integration**: Natural language explanations
4. **K8s Deployment**: Production infrastructure

---

**Author**: Aleksandr Plotnikov  
**Date**: November 10, 2025  
**Version**: 0.1.0
