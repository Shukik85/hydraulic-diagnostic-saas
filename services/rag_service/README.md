# RAG Service: DeepSeek-R1 Self-hosted

🤖 **Reasoning AI для интерпретации GNN результатов**

## Архитектура

```
┌─────────────────────────────────┐
│    Diagnosis Service            │
│  (Orchestrator)                 │
└────────────┬────────────────────┘
             │
             ↓ gRPC
┌─────────────────────────────────┐
│    GNN Service                  │
│  (ML Inference)                 │
└────────────┬────────────────────┘
             │
             ↓ Results
┌─────────────────────────────────┐
│    RAG Service                  │
│  DeepSeek-R1-Distill-32B        │
│  • Reasoning интерпретация      │
│  • Понятные объяснения          │
│  • Приоритизированные рекомендации│
└─────────────────────────────────┘
```

## Характеристики

### Модель
- **Model**: DeepSeek-R1-Distill-Qwen-32B
- **Size**: ~80GB
- **GPUs**: 2x A100 (80GB each)
- **Inference Engine**: vLLM с tensor parallelism
- **Latency**: ~2-3 секунды per request
- **Throughput**: ~10-15 requests/minute

### Features
- ✅ Reasoning из коробки (Chain-of-Thought)
- ✅ Интерпретация GNN outputs
- ✅ Контекстные объяснения
- ✅ Приоритизированные рекомендации
- ✅ Прогнозирование отказов
- ✅ Multi-GPU support
- ✅ Production-ready с vLLM

## Quick Start

### Local Development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model
python download_model.py

# 3. Run service
export TENSOR_PARALLEL_SIZE=2
python main.py
```

### Docker

```bash
# Build
docker build -t rag-service:latest .

# Run with GPU
docker run --gpus '"device=0,1"' \
  -p 8004:8004 \
  -v $(pwd)/models:/app/models \
  rag-service:latest
```

### Kubernetes

```bash
# Deploy
kubectl apply -f kubernetes/deployment.yaml

# Check status
kubectl get pods -n hydraulic-prod -l app=rag-service

# Logs
kubectl logs -f deployment/rag-service -n hydraulic-prod
```

## API Usage

### 1. Interpret GNN Diagnosis

```python
import requests

response = requests.post(
    "http://rag-service:8004/interpret/diagnosis",
    json={
        "gnn_result": {
            "overall_health_score": 0.65,
            "anomalies": [
                {
                    "anomaly_type": "pressure_drop",
                    "severity": "high",
                    "confidence": 0.85,
                    "affected_components": ["main_pump"]
                }
            ],
            "component_health": [
                {
                    "component_id": "pump_001",
                    "component_type": "Main Pump",
                    "health_score": 0.65,
                    "degradation_rate": 0.08
                }
            ]
        },
        "equipment_context": {
            "equipment_id": "exc_001",
            "equipment_type": "Excavator",
            "model": "CAT-320D",
            "manufacturer": "Caterpillar",
            "operating_hours": 8500
        }
    }
)

result = response.json()
print(result["summary"])     # Понятное резюме
print(result["reasoning"])   # Reasoning steps
print(result["recommendations"])  # Приоритизированные действия
```

### 2. Explain Anomaly

```python
response = requests.post(
    "http://rag-service:8004/explain/anomaly",
    json={
        "anomaly_type": "pressure_drop",
        "context": {
            "component": "main_pump",
            "current_pressure": 115.3,
            "normal_pressure": 150.0,
            "operating_hours": 8500
        }
    }
)

print(response.json()["explanation"])
```

### 3. Generic Generation

```python
response = requests.post(
    "http://rag-service:8004/generate",
    json={
        "prompt": "Explain hydraulic pump failure modes",
        "max_tokens": 1024,
        "temperature": 0.7
    }
)

print(response.json()["response"])
```

## Performance

### GPU Utilization

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Expected:
# GPU 0: ~85% utilization, ~70GB memory
# GPU 1: ~85% utilization, ~70GB memory
```

### Latency

| Request Type | Latency (p50) | Latency (p99) |
|--------------|---------------|---------------|
| Diagnosis Interpretation | 2.1s | 3.5s |
| Anomaly Explanation | 1.5s | 2.8s |
| Generic Generation | 1.8s | 3.2s |

### Throughput

- **Sequential**: ~15-20 req/min
- **Batched** (vLLM): ~40-50 req/min

## Monitoring

### Health Checks

```bash
# Health
curl http://rag-service:8004/health

# Readiness
curl http://rag-service:8004/ready
```

### Metrics

Prometheus metrics available at `/metrics`:
- `rag_requests_total` - Total requests
- `rag_request_duration_seconds` - Request latency
- `rag_gpu_utilization_percent` - GPU usage
- `rag_gpu_memory_used_bytes` - GPU memory

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Model ID |
| `TENSOR_PARALLEL_SIZE` | `2` | Number of GPUs |
| `GPU_MEMORY_UTIL` | `0.90` | GPU memory utilization |
| `MAX_MODEL_LEN` | `8192` | Max sequence length |
| `PORT` | `8004` | HTTP port |
| `LOG_LEVEL` | `INFO` | Logging level |

## Troubleshooting

### Model Not Loading

```bash
# Check GPU availability
nvidia-smi

# Check disk space
df -h /app/models

# Re-download model
python download_model.py
```

### OOM (Out of Memory)

```bash
# Reduce GPU memory utilization
export GPU_MEMORY_UTIL=0.85

# Reduce max sequence length
export MAX_MODEL_LEN=4096
```

### Slow Inference

```bash
# Check GPU utilization
nvidia-smi

# Enable KV cache
export VLLM_USE_KV_CACHE=1

# Increase batch size
export VLLM_MAX_NUM_SEQS=8
```

## Production Checklist

- [ ] Model downloaded and cached
- [ ] 2x A100 GPUs available
- [ ] vLLM installed correctly
- [ ] Health checks passing
- [ ] Latency < 5s p99
- [ ] GPU utilization > 70%
- [ ] Monitoring configured
- [ ] Alerts setup
- [ ] Backup inference endpoint

## License

Proprietary - Hydraulic Diagnostic SaaS
