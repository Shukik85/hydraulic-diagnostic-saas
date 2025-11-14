# Monitoring & Admin Endpoints Documentation

## 📊 Overview

Все микросервисы теперь имеют стандартизированные monitoring и admin endpoints.

---

## 🏥 Health Endpoints

### `GET /health`

**Назначение**: Liveness probe для Kubernetes  
**Access**: Public (no auth)  
**Latency**: < 100ms  

**Response**:
```json
{
  "service": "rag-service",
  "status": "healthy",
  "timestamp": "2025-11-13T06:45:00Z",
  "checks": {
    "database": "ok",
    "redis": "ok",
    "disk": "ok",
    "memory": "ok"
  }
}
```

**Status values**:
- `healthy` - All systems operational ✅
- `degraded` - Non-critical issues ⚠️
- `unhealthy` - Critical failures ❌

**Available on**:
- Equipment Service: `http://equipment-service:8002/health`
- Diagnosis Service: `http://diagnosis-service:8003/health`
- GNN Service: `http://gnn-service:8002/health`
- RAG Service: `http://rag-service:8004/health`

---

### `GET /ready`

**Назначение**: Readiness probe для Kubernetes  
**Access**: Public  

**Response**:
```json
{
  "status": "ready"
}
```

**Returns**:
- `200` - Service ready to accept traffic
- `503` - Service not ready (starting up)

---

### `GET /metrics`

**Назначение**: Prometheus metrics export  
**Access**: Public (но обычно internal)  
**Content-Type**: `text/plain; version=0.0.4`  

**Response** (Prometheus format):
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{service="gnn-service",method="POST",endpoint="/inference",status="200"} 1547

# HELP http_request_duration_seconds HTTP request latency
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{service="gnn-service",method="POST",endpoint="/inference",le="0.5"} 1234

# HELP ml_inference_total Total ML inferences
# TYPE ml_inference_total counter
ml_inference_total{service="gnn-service",model="universal_gnn_v1",status="success"} 1502

# HELP cpu_usage_percent CPU usage percentage
# TYPE cpu_usage_percent gauge
cpu_usage_percent{service="gnn-service"} 45.2

# HELP memory_usage_bytes Memory usage in bytes
# TYPE memory_usage_bytes gauge
memory_usage_bytes{service="gnn-service"} 4294967296
```

**Metrics Categories**:
1. **HTTP Requests**: request counts, latency, status codes
2. **ML Operations**: inference counts, latency, success rates
3. **Resources**: CPU, memory, GPU, disk
4. **Service Health**: uptime, error rates

---

## 🔧 Admin Endpoints

### GNN Service Admin

#### `POST /admin/model/deploy`

**Назначение**: Deploy новой GNN модели  
**Access**: Admin only (JWT + `is_admin=true`)  

**Request**:
```json
{
  "model_path": "/models/universal_gnn_v2.onnx",
  "version": "2.0.1",
  "description": "Retrained with Nov-2025 data",
  "validate_first": true
}
```

**Response**:
```json
{
  "status": "success",
  "deployment_id": "dpl_20251113_064500",
  "model_version": "2.0.1",
  "deployed_at": "2025-11-13T06:45:00Z",
  "validation_results": {
    "valid": true,
    "input_shape": [1, 10, 32],
    "output_shape": [1, 3]
  },
  "message": "Model 2.0.1 deployed. Restart inference workers to apply."
}
```

**Process**:
1. Validate model file
2. (Optional) Run validation tests
3. Backup current model
4. Copy to production path
5. Update metadata
6. Reload workers (manual restart or K8s rolling update)

---

#### `POST /admin/model/rollback`

**Назначение**: Откат к предыдущей версии  
**Access**: Admin only  

**Request**:
```json
{
  "backup_filename": "model_20251112_120000.onnx"
}
```

---

#### `GET /admin/model/info`

**Назначение**: Информация о текущей модели  
**Access**: Admin only  

**Response**:
```json
{
  "model_version": "2.0.1",
  "model_path": "/app/models/current/model.onnx",
  "deployed_at": "2025-11-13T06:45:00Z",
  "model_size_mb": 156.8,
  "input_shape": [1, 10, 32],
  "output_classes": 3,
  "framework": "onnx"
}
```

---

#### `POST /admin/training/start`

**Назначение**: Запустить обучение GNN  
**Access**: Admin only  

**Request**:
```json
{
  "dataset_path": "/data/gnn_graphs_multilabel.pt",
  "config": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "max_epochs": 100,
    "early_stopping_patience": 10
  },
  "experiment_name": "universal_gnn_v2_training"
}
```

**Response**:
```json
{
  "job_id": "train_20251113_064500",
  "status": "started",
  "started_at": "2025-11-13T06:45:00Z",
  "tensorboard_url": "http://tensorboard:6006/#scalars&run=train_20251113_064500"
}
```

---

#### `GET /admin/training/{job_id}/status`

**Назначение**: Статус обучения  
**Access**: Admin only  

**Response**:
```json
{
  "job_id": "train_20251113_064500",
  "status": "running",
  "progress": {
    "current_epoch": 45,
    "max_epochs": 100,
    "progress_percent": 45.0
  },
  "metrics": {
    "train_loss": 0.1234,
    "val_loss": 0.0987,
    "best_val_loss": 0.0952,
    "val_accuracy": 0.9234
  },
  "started_at": "2025-11-13T06:45:00Z",
  "eta_minutes": 25
}
```

---

### RAG Service Admin

#### `GET /admin/config`

**Назначение**: Получить текущую RAG конфигурацию  
**Access**: Admin only  

**Response**:
```json
{
  "config": {
    "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 2048,
    "stop_sequences": ["</думает>", "<|end|>"],
    "system_prompt": "Ты эксперт по диагностике..."
  },
  "version": 3,
  "updated_at": "2025-11-13T06:45:00Z",
  "updated_by": "admin@company.com"
}
```

---

#### `PUT /admin/config`

**Назначение**: Обновить RAG конфигурацию  
**Access**: Admin only  

**Request**:
```json
{
  "temperature": 0.6,
  "max_tokens": 1536,
  "system_prompt": "Updated system prompt..."
}
```

**Response**: Same as GET

**Note**: Changes apply immediately to new requests.

---

#### `GET /admin/config/history`

**Назначение**: История изменений конфигурации  
**Access**: Admin only  

**Response**:
```json
{
  "history": [
    {
      "filename": "config_20251113_064500_admin.json",
      "version": 3,
      "updated_at": "2025-11-13T06:45:00Z",
      "updated_by": "admin@company.com"
    },
    {
      "filename": "config_20251112_120000_john.json",
      "version": 2,
      "updated_at": "2025-11-12T12:00:00Z",
      "updated_by": "john@company.com"
    }
  ]
}
```

---

#### `POST /admin/prompt/test`

**Назначение**: Тестирование prompt template  
**Access**: Admin only  

**Request**:
```json
{
  "template": {
    "name": "diagnosis_v2",
    "category": "diagnosis",
    "template_text": "Проанализируй: {{gnn_results}}",
    "variables": ["gnn_results"],
    "language": "ru"
  },
  "test_data": {
    "gnn_results": "{...sample data...}"
  }
}
```

**Response**:
```json
{
  "rendered_prompt": "Проанализируй: {...}",
  "response": "<думает>...analysis...</думает> Рекомендации: ...",
  "tokens_used": 847
}
```

---

## 🔐 Authentication

### Admin Endpoints

**Require**: JWT token с `role: admin` или `role: superadmin`

**Headers**:
```
Authorization: Bearer <admin_jwt_token>
```

**Error Responses**:
- `401 Unauthorized` - Invalid/expired token
- `403 Forbidden` - Not admin role

### Monitoring Endpoints

**Public** (no auth required):
- `/health`
- `/ready`
- `/metrics`

**Note**: В production рекомендуется ограничить `/metrics` через network policies.

---

## 📊 Prometheus Integration

### Scrape Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'hydraulic-services'
    scrape_interval: 15s
    static_configs:
      - targets:
          - 'equipment-service:8002'
          - 'diagnosis-service:8003'
          - 'gnn-service:8002'
          - 'rag-service:8004'
    metrics_path: '/metrics'
```

### Key Metrics to Monitor

**Service Health**:
- `service_up` - Service availability (0 or 1)
- `http_requests_total` - Request counts
- `http_request_duration_seconds` - Latency

**ML Operations**:
- `ml_inference_total` - Inference counts
- `ml_inference_duration_seconds` - Inference latency
- `gpu_memory_usage_bytes` - GPU memory

**Resources**:
- `cpu_usage_percent` - CPU utilization
- `memory_usage_bytes` - Memory usage
- `database_connections_active` - DB pool

### Grafana Dashboards

Import pre-built dashboard:
```bash
kubectl apply -f infrastructure/monitoring/grafana-dashboards.yaml
```

---

## 🚨 Alerting

### Example Prometheus Alerts

```yaml
groups:
  - name: services
    rules:
      - alert: ServiceDown
        expr: service_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.service }} is down"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency on {{ $labels.service }}"
      
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.service }}"
```

---

## 🔧 Usage Examples

### Health Check

```bash
# Check all services
for port in 8002 8003 8004; do
  curl http://localhost:${port}/health | jq '.status'
done
```

### Deploy GNN Model

```bash
curl -X POST http://gnn-service:8002/admin/model/deploy \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/universal_gnn_v2.onnx",
    "version": "2.0.1",
    "validate_first": true
  }'
```

### Update RAG Config

```bash
curl -X PUT http://rag-service:8004/admin/config \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 0.6,
    "max_tokens": 1536,
    "system_prompt": "Updated prompt..."
  }'
```

### View Metrics

```bash
# Raw Prometheus metrics
curl http://rag-service:8004/metrics

# Parse specific metric
curl -s http://rag-service:8004/metrics | grep ml_inference_total
```

---

## 🏗️ Implementation Checklist

### Per Service

- [ ] **Equipment Service**
  - [x] `/health` endpoint
  - [x] `/metrics` endpoint
  - [x] `/ready` endpoint
  - [x] Prometheus middleware

- [ ] **Diagnosis Service**
  - [x] `/health` endpoint (with GNN/RAG checks)
  - [x] `/metrics` endpoint
  - [x] `/ready` endpoint

- [ ] **GNN Service**
  - [x] `/health` endpoint
  - [x] `/metrics` endpoint
  - [x] `/admin/model/deploy` endpoint
  - [x] `/admin/model/rollback` endpoint
  - [x] `/admin/training/start` endpoint
  - [x] `/admin/training/{id}/status` endpoint

- [ ] **RAG Service**
  - [x] `/health` endpoint
  - [x] `/metrics` endpoint
  - [x] `/admin/config` GET/PUT
  - [x] `/admin/config/history` GET
  - [x] `/admin/prompt/test` POST

### Infrastructure

- [ ] Prometheus deployment
- [ ] Grafana dashboards
- [ ] Alert rules configured
- [ ] Network policies for `/metrics`

---

## ✅ Acceptance Criteria

- [ ] All services respond to `/health` < 100ms
- [ ] `/metrics` returns valid Prometheus format
- [ ] Admin endpoints require authentication
- [ ] Model deployment works end-to-end
- [ ] RAG config updates apply immediately
- [ ] Prometheus scrapes all services
- [ ] Grafana dashboards show data
- [ ] Alerts fire correctly

---

## 🚀 Next Steps

1. Apply changes to all services
2. Test endpoints locally
3. Deploy to staging
4. Configure Prometheus/Grafana
5. Set up alerts
6. Update OpenAPI specs
7. Generate TypeScript client
8. Integrate in Django Admin (optional)
