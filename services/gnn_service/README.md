# GNN Service - Production-Ready

🚀 **Universal Temporal GNN (GAT + LSTM)** для multi-label classification состояния компонентов гидравлических систем.

## 🎯 Статус: Production-Ready

✅ Модульная архитектура  
✅ OpenTelemetry distributed tracing  
✅ Rate limiting (token bucket)  
✅ Dynamic batching  
✅ Multi-model A/B testing  
✅ Kubernetes-ready (/healthz, /readyz)  
✅ Security hardening  
✅ Full observability  

---

## 🏛️ Architecture

```
┌────────────────────────────────────────┐
│  FastAPI Application (main.py)         │
│  ├─ Request ID Middleware                 │
│  ├─ OpenTelemetry Middleware              │
│  ├─ Rate Limit Middleware (100 req/60s)  │
│  ├─ Body Size Limit (10MB)                │
│  └─ CORS Middleware                       │
└────────────────┬────────────────────────┘
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
     ┌───────────┼───────────┐
     │            │            │
┌────▼────┐  ┌────▼─────┐  ┌▼──────────┐
│ PyTorch  │  │ TimescaleDB│  │ Prometheus│
│ Model    │  │ (Sensors)  │  │ Metrics    │
└──────────┘  └───────────┘  └────────────┘
```

### 📦 Modular Structure

```
services/gnn_service/
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   └── validators.py           # Request validation
│   ├── inference/
│   │   ├── __init__.py             # Public API
│   │   ├── inference_engine.py     # Main engine (550 lines)
│   │   ├── exceptions.py           # All exceptions
│   │   ├── metrics.py              # Prometheus metrics
│   │   ├── validation.py           # Tensor validation
│   │   ├── model_registry.py       # A/B testing
│   │   ├── cache.py                # Topology cache + stampede protection
│   │   ├── batching.py             # Dynamic batching
│   │   ├── request_context.py      # X-Request-ID tracking
│   │   ├── model_manager.py        # Model loading
│   │   └── dynamic_graph_builder.py # Polars-native graph building
│   ├── middleware/
│   │   ├── __init__.py             # Exports
│   │   ├── opentelemetry.py        # Distributed tracing
│   │   └── rate_limiter.py         # Token bucket rate limiting
│   ├── data/                   # Feature engineering
│   ├── schemas/                # Pydantic models
│   └── services/               # Topology service
├── configs/
│   └── config.py               # Configuration
├── requirements.txt         # Dependencies
└── .env.example             # Environment template
```

---

## 🚀 Production Features

### 1. 🔍 OpenTelemetry Distributed Tracing

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

### 2. ⏱️ Rate Limiting (Token Bucket)

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

### 3. 📦 Dynamic Batching

```python
# Automatic batching with configurable window
# Max batch size: 32 (configurable)
# Max wait time: 50ms (configurable)
# Race condition protection
```

**Configuration:**
```python
config = InferenceConfig(
    enable_dynamic_batching=True,
    batch_size=32,
    max_wait_ms=50.0
)
```

### 4. 🎯 Multi-Model A/B Testing

```python
# Traffic splitting with consistent hashing
# Example: v1 (80%) vs v2 (20%)
config = InferenceConfig(
    model_versions={
        "v1": ModelConfig(path="v1.ckpt", traffic=0.8),
        "v2": ModelConfig(path="v2.ckpt", traffic=0.2)
    }
)
```

### 5. 💾 Topology Caching + Stampede Protection

```python
# LRU cache with TTL: 100 items, 300s TTL
# AsyncTopologyCache with stampede protection
# Prevents duplicate topology builds
```

### 6. 🏷️ Request ID Propagation

```python
# X-Request-ID header tracking through entire pipeline
# Automatic generation if not provided
# Available in all logs and traces
```

### 7. ☸️ Kubernetes Health Endpoints

```yaml
# Liveness probe
GET /healthz -> {"status": "ok"}

# Readiness probe
GET /readyz -> {"ready": true, "components": {...}}
```

**Kubernetes manifest:**
```yaml
livenessProbe:
  httpGet:
    path: /healthz
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /readyz
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### 8. 🔒 Security Hardening

✅ **Request size limiting:** 10MB max  
✅ **CORS strict origins:** Whitelist only  
✅ **Database timeout:** 10s (prevents hangs)  
✅ **Inference timeout:** 30s (configurable)  
✅ **RCE prevention:** `weights_only=True` in torch.load  

### 9. ⚡ Performance Optimizations

✅ **Polars instead of pandas:** Async-friendly, no GIL blocking  
✅ **TaskGroup (Python 3.11+):** Better async task management  
✅ **torch.compile:** JIT compilation (if enabled)  
✅ **CPU fallback:** Graceful degradation on GPU OOM  

---

## 🛠️ Configuration

### Environment Variables

```bash
# Service
PORT=8000
HOST=0.0.0.0

# Inference
MODEL_PATH=models/v2.0.0.ckpt
DEVICE=auto  # cpu, cuda, auto
BATCH_SIZE=32
ENABLE_DYNAMIC_BATCHING=true
INFERENCE_TIMEOUT_S=30

# Database
TIMESCALEDB_URL=postgresql://user:pass@localhost:5432/hydraulic
DB_QUERY_TIMEOUT_S=10

# OpenTelemetry
OTEL_ENABLED=true
OTEL_SERVICE_NAME=gnn-service
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_S=60
REDIS_URL=redis://localhost:6379

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
gnn_inference_requests_total{model_version, status}
gnn_inference_errors_total{error_type}

# Latency
gnn_inference_duration_seconds{model_version}

# Batching
gnn_inference_batch_size
gnn_request_queue_size

# Cache
gnn_topology_cache_size

# GPU
gnn_model_gpu_memory_bytes{model_version, device}
```

### OpenTelemetry Spans

```
POST /v1/diagnose
  ├─ validate_request
  ├─ get_topology (cache: hit/miss)
  ├─ build_graph
  │   ├─ fetch_sensor_data (TimescaleDB)
  │   ├─ create_node_features
  │   └─ create_edge_features
  ├─ validate_tensor
  ├─ inference
  │   └─ model_forward
  └─ postprocess
```

### Grafana Dashboard

```json
{
  "panels": [
    {"title": "Request Rate", "metric": "rate(gnn_inference_requests_total[5m])"},
    {"title": "Error Rate", "metric": "rate(gnn_inference_errors_total[5m])"},
    {"title": "P95 Latency", "metric": "histogram_quantile(0.95, gnn_inference_duration_seconds)"},
    {"title": "GPU Memory", "metric": "gnn_model_gpu_memory_bytes"},
    {"title": "Queue Size", "metric": "gnn_request_queue_size"},
    {"title": "Cache Hit Rate", "metric": "rate(topology_cache_hits) / rate(topology_cache_total)"}
  ]
}
```

---

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Expose port
EXPOSE 8000

# Run
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gnn-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gnn-service
  template:
    metadata:
      labels:
        app: gnn-service
    spec:
      containers:
      - name: gnn-service
        image: your-registry/gnn-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:4318"
        - name: REDIS_URL
          value: "redis://redis:6379"
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "2"
            memory: "4Gi"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: gnn-service
spec:
  selector:
    app: gnn-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 📝 API Examples

### Diagnose Equipment

```bash
curl -X POST http://localhost:8000/v1/diagnose \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: req-123" \
  -d '{
    "equipment_id": "excavator_001",
    "topology_id": "double_pump_v1",
    "timestamp": "2025-12-17T01:00:00Z",
    "sensor_readings": {
      "pump_1": {"pressure": 150.5, "temperature": 65.2},
      "valve_1": {"position": 0.75, "leakage": 0.01}
    }
  }'
```

**Response:**
```json
{
  "status": "success",
  "equipment_id": "excavator_001",
  "timestamp": "2025-12-17T01:00:00Z",
  "diagnosis": {
    "health": {"score": 0.85},
    "degradation": {"rate": 0.12},
    "anomaly": {
      "predictions": {
        "pressure_drop": 0.05,
        "overheating": 0.15,
        "cavitation": 0.02,
        "leakage": 0.08
      }
    },
    "inference_time_ms": 42.3
  }
}
```

### Check Health

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}

curl http://localhost:8000/readyz
# {"ready": true, "components": {"inference_engine": "ok"}}

curl http://localhost:8000/metrics
# {"status": "ok", "inference_engine": {...}}
```

---

## 👥 Development

### Local Setup

```bash
# Clone repo
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas/services/gnn_service

# Create venv
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy env template
cp .env.example .env

# Run service
uvicorn src.api.main:app --reload
```

### Testing

```bash
# Unit tests
pytest tests/unit

# Integration tests
pytest tests/integration

# Load tests
locust -f tests/load/locustfile.py
```

---

## 🛡️ Troubleshooting

### GPU Out of Memory

```python
# Enable CPU fallback
config = InferenceConfig(
    device="cuda",
    fallback_to_cpu=True
)
```

### Rate Limit Exceeded

```bash
# Increase limits
RATE_LIMIT_REQUESTS=200
RATE_LIMIT_WINDOW_S=60

# Or disable
RATE_LIMIT_ENABLED=false
```

### Database Timeout

```bash
# Increase timeout
DB_QUERY_TIMEOUT_S=30
```

### Model Loading Fails

```bash
# Check checkpoint compatibility
# Re-save with PyTorch 2.0+ using weights_only=True
torch.save({
    'model_state_dict': model.state_dict(),
    'normalizer_stats': normalizer.get_stats()
}, 'model.ckpt')
```

---

## 📚 References

- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenTelemetry](https://opentelemetry.io/)
- [Prometheus](https://prometheus.io/)
- [Kubernetes Probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)

---

## 🎉 Credits

**Built with:**
- PyTorch 2.5+ & PyTorch Geometric 2.7+
- FastAPI 0.115+
- Polars 0.20+
- OpenTelemetry 1.28+
- Python 3.11+ (3.14+ recommended)

**Architecture by:** Senior ML Engineer @ Hydraulic Diagnostics Team

---

## 📝 License

MIT License - see LICENSE file for details
