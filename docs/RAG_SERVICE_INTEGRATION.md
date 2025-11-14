# RAG Service Integration Guide

## Overview

RAG Service добавляет **reasoning-based interpretation** для результатов GNN модели, используя DeepSeek-R1-Distill-32B.

## Architecture Integration

### Flow

```
User Request
    ↓
Diagnosis Service (Orchestrator)
    ↓
1. Query TimescaleDB (sensor data)
    ↓
2. Call GNN Service (ML inference)
    ↓  
    GNN Results:
    - health_scores: [0.65, 0.82, 0.45]
    - anomalies: ["pressure_drop"]
    - degradation_rates: [0.08, 0.02, 0.12]
    ↓
3. Call RAG Service (interpretation)
    ↓
    RAG Output:
    - summary: "Критическое падение давления в насосе..."
    - reasoning: "<думает>Анализирую..."
    - recommendations: ["Замените фильтр", ...]
    - prognosis: "Отказ через 8-10 дней"
    ↓
4. Return to Frontend
    ↓
User sees понятное объяснение!
```

## Integration Points

### 1. Diagnosis Service → RAG Service

**File**: `services/diagnosis_service/rag_client.py`

```python
import grpc
import requests
from typing import Dict

class RAGClient:
    """Client для RAG Service."""
    
    def __init__(self, rag_url: str = "http://rag-service:8004"):
        self.rag_url = rag_url
    
    async def interpret_diagnosis(
        self,
        gnn_result: Dict,
        equipment_context: Dict
    ) -> Dict:
        """Interpret GNN results."""
        
        response = requests.post(
            f"{self.rag_url}/interpret/diagnosis",
            json={
                "gnn_result": gnn_result,
                "equipment_context": equipment_context
            },
            timeout=30  # RAG может быть медленнее
        )
        
        response.raise_for_status()
        return response.json()
```

**Usage in Diagnosis Service**:

```python
# services/diagnosis_service/main.py

@app.post("/diagnosis")
async def run_diagnosis(request: DiagnosisRequest):
    # 1. Get sensor data
    sensor_data = await get_sensor_data(...)
    
    # 2. Run GNN inference
    gnn_result = await gnn_service.predict(sensor_data)
    
    # 3. Interpret with RAG (NEW!)
    rag_client = RAGClient()
    interpretation = await rag_client.interpret_diagnosis(
        gnn_result=gnn_result,
        equipment_context={
            "equipment_id": request.equipment_id,
            "equipment_type": equipment.type,
            "model": equipment.model,
            # ... metadata
        }
    )
    
    # 4. Combine results
    return {
        "gnn_results": gnn_result,
        "interpretation": interpretation,  # NEW!
        "timestamp": datetime.utcnow().isoformat()
    }
```

### 2. Frontend Display

**Component**: `services/frontend/components/DiagnosisResults.vue`

```vue
<template>
  <div class="diagnosis-results">
    <!-- Summary Section -->
    <div class="summary-card">
      <h3>📊 Результаты диагностики</h3>
      <p class="summary-text">{{ interpretation.summary }}</p>
      
      <!-- Health Score -->
      <div class="health-indicator">
        <CircularProgress :value="overallHealth" />
        <span>{{ overallHealth }}% здоровье системы</span>
      </div>
    </div>
    
    <!-- Reasoning Section (Expandable) -->
    <div class="reasoning-card" v-if="interpretation.reasoning">
      <button @click="showReasoning = !showReasoning">
        🧠 Процесс анализа (reasoning)
        <ChevronIcon :expanded="showReasoning" />
      </button>
      
      <div v-show="showReasoning" class="reasoning-content">
        <pre>{{ interpretation.reasoning }}</pre>
      </div>
    </div>
    
    <!-- Recommendations -->
    <div class="recommendations-card">
      <h4>✅ Рекомендации</h4>
      <ul>
        <li 
          v-for="(rec, idx) in interpretation.recommendations" 
          :key="idx"
          class="recommendation-item"
        >
          <span class="priority">{{ getPriority(rec) }}</span>
          <span class="text">{{ rec }}</span>
        </li>
      </ul>
    </div>
    
    <!-- Prognosis -->
    <div class="prognosis-card" v-if="interpretation.prognosis">
      <h4>🔮 Прогноз</h4>
      <p>{{ interpretation.prognosis }}</p>
    </div>
    
    <!-- Technical Details (Collapsible) -->
    <div class="technical-details">
      <button @click="showTechnical = !showTechnical">
        🔧 Технические детали
      </button>
      
      <div v-show="showTechnical">
        <!-- GNN raw outputs -->
        <ComponentHealthTable :components="gnnResults.component_health" />
        <AnomaliesTable :anomalies="gnnResults.anomalies" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Props {
  gnnResults: any
  interpretation: {
    summary: string
    reasoning: string
    recommendations: string[]
    prognosis: string
  }
}

const props = defineProps<Props>()
const showReasoning = ref(false)
const showTechnical = ref(false)

const overallHealth = computed(() => {
  return Math.round(props.gnnResults.overall_health_score * 100)
})

function getPriority(rec: string): string {
  if (rec.toLowerCase().includes('срочно') || rec.toLowerCase().includes('критично')) {
    return '🔴 Высокий'
  } else if (rec.toLowerCase().includes('рекомендуется')) {
    return '🟡 Средний'
  }
  return '🟢 Низкий'
}
</script>
```

## Deployment

### 1. Update Docker Compose

```yaml
# docker-compose.yml
services:
  # ... existing services
  
  rag-service:
    build:
      context: ./services/rag_service
      dockerfile: Dockerfile
    container_name: rag-service
    ports:
      - "8004:8004"
    environment:
      - TENSOR_PARALLEL_SIZE=2
      - GPU_MEMORY_UTIL=0.90
      - MAX_MODEL_LEN=8192
    volumes:
      - ./models:/app/models
      - ./knowledge_base:/app/knowledge_base
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0', '1']
              capabilities: [gpu]
    depends_on:
      - gnn-service
    networks:
      - hydraulic-network
```

### 2. Update Kubernetes

```bash
# Deploy RAG service
kubectl apply -f services/rag_service/kubernetes/deployment.yaml

# Verify
kubectl get pods -n hydraulic-prod -l app=rag-service

# Check logs
kubectl logs -f deployment/rag-service -n hydraulic-prod
```

### 3. Update Service Mesh

```yaml
# infrastructure/istio/virtual-services.yaml
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: rag-service-vs
  namespace: hydraulic-prod
spec:
  hosts:
    - rag-service.hydraulic-prod.svc.cluster.local
  http:
    - route:
        - destination:
            host: rag-service.hydraulic-prod.svc.cluster.local
            port:
              number: 8004
      timeout: 60s  # RAG может быть медленнее
      retries:
        attempts: 1  # No retries для expensive operations
```

## Performance Optimization

### 1. Caching Interpretations

```python
# services/diagnosis_service/cache.py
import redis
import json
import hashlib

class InterpretationCache:
    """Cache RAG interpretations."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour
    
    def get_cache_key(self, gnn_result: dict) -> str:
        """Generate cache key from GNN result."""
        # Hash important parts
        cache_data = {
            "health": gnn_result["overall_health_score"],
            "anomalies": [a["anomaly_type"] for a in gnn_result["anomalies"]]
        }
        return hashlib.sha256(
            json.dumps(cache_data, sort_keys=True).encode()
        ).hexdigest()
    
    async def get(self, gnn_result: dict) -> Optional[dict]:
        """Get cached interpretation."""
        key = self.get_cache_key(gnn_result)
        cached = self.redis.get(f"rag:interpretation:{key}")
        if cached:
            return json.loads(cached)
        return None
    
    async def set(self, gnn_result: dict, interpretation: dict):
        """Cache interpretation."""
        key = self.get_cache_key(gnn_result)
        self.redis.setex(
            f"rag:interpretation:{key}",
            self.ttl,
            json.dumps(interpretation)
        )
```

### 2. Async Processing

```python
# services/diagnosis_service/async_rag.py
from celery import Celery

celery_app = Celery('rag_tasks', broker='redis://redis:6379/0')

@celery_app.task
def interpret_diagnosis_async(diagnosis_id: str, gnn_result: dict, context: dict):
    """Асинхронная интерпретация (не блокирует ответ)."""
    rag_client = RAGClient()
    interpretation = rag_client.interpret_diagnosis(gnn_result, context)
    
    # Save to database
    save_interpretation(diagnosis_id, interpretation)
    
    # Notify frontend via WebSocket
    notify_frontend(diagnosis_id, interpretation)
    
    return interpretation
```

## Monitoring

### Metrics

```python
# services/rag_service/metrics.py
from prometheus_client import Counter, Histogram, Gauge

RAG_REQUESTS = Counter(
    'rag_requests_total',
    'Total RAG requests',
    ['endpoint', 'status']
)

RAG_LATENCY = Histogram(
    'rag_request_duration_seconds',
    'RAG request latency',
    ['endpoint'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

RAG_GPU_UTIL = Gauge(
    'rag_gpu_utilization_percent',
    'GPU utilization',
    ['gpu_id']
)
```

### Alerts

```yaml
# infrastructure/monitoring/rag-alerts.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-alerts
data:
  alerts.yaml: |
    groups:
      - name: rag-service
        rules:
          - alert: RAGHighLatency
            expr: histogram_quantile(0.99, rag_request_duration_seconds) > 10
            for: 5m
            labels:
              severity: warning
            annotations:
              summary: "RAG service high latency"
              
          - alert: RAGLowGPUUtilization
            expr: avg(rag_gpu_utilization_percent) < 30
            for: 10m
            labels:
              severity: info
            annotations:
              summary: "RAG GPUs underutilized"
              
          - alert: RAGServiceDown
            expr: up{job="rag-service"} == 0
            for: 1m
            labels:
              severity: critical
            annotations:
              summary: "RAG service is down"
```

## Testing

### Unit Tests

```python
# services/rag_service/tests/test_interpreter.py
import pytest
from gnn_interpreter import GNNInterpreter

@pytest.fixture
def interpreter():
    return GNNInterpreter()

def test_interpret_diagnosis(interpreter):
    gnn_result = {
        "overall_health_score": 0.65,
        "anomalies": [
            {
                "anomaly_type": "pressure_drop",
                "severity": "high",
                "confidence": 0.85
            }
        ]
    }
    
    context = {
        "equipment_id": "exc_001",
        "equipment_type": "Excavator"
    }
    
    result = interpreter.interpret_diagnosis(gnn_result, context)
    
    assert "summary" in result
    assert "reasoning" in result
    assert len(result["recommendations"]) > 0
```

### Integration Tests

```python
# tests/integration/test_diagnosis_with_rag.py
import pytest
import requests

def test_full_diagnosis_flow():
    # 1. Run diagnosis
    response = requests.post(
        "http://diagnosis-service:8003/diagnosis",
        json={
            "equipment_id": "exc_001",
            "time_window": {
                "start": "2025-11-01T00:00:00Z",
                "end": "2025-11-13T00:00:00Z"
            }
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    
    # 2. Check interpretation present
    assert "interpretation" in result
    assert "summary" in result["interpretation"]
    assert "recommendations" in result["interpretation"]
    
    # 3. Verify reasoning quality
    assert len(result["interpretation"]["summary"]) > 50
    assert len(result["interpretation"]["recommendations"]) > 0
```

## Troubleshooting

See `services/rag_service/README.md` для detailed troubleshooting guide.

## Next Steps

1. ✅ Deploy RAG Service
2. ⏳ Integrate with Diagnosis Service
3. ⏳ Update Frontend components
4. ⏳ Load testing
5. ⏳ Production deployment
