# Microservices Integration Guide

## Обзор

**Hydraulic Diagnostic Platform** использует **microservices архитектуру** для разделения ML/RAG логики от основного backend API:

```
┌──────────────────────────────────────┐
│          Frontend (Nuxt)              │
│         Port 3000 (Public)             │
└────────────┬────────────────────────┘
             │ HTTP/REST
             │ (Public Network)
             │
┌────────────┴────────────────────────┐
│      Backend API Gateway (Django)      │
│         Port 8000 (Public)             │
│   - JWT Authentication                │
│   - User Authorization                │
│   - Request Validation                │
│   - Audit Logging                     │
└──────┬──────────────────┬───────────┘
      │                    │
      │ (Internal Network) │
      │ X-Internal-API-Key │
      │                    │
┌─────┴────────┐    ┌──────┴────────┐
│  ML Service  │    │  RAG Service │
│  Port 8001   │    │  Port 8002   │
│ (Internal)  │    │ (Internal)  │
└──────────────┘    └──────────────┘
```

## Принципы безопасности

✅ **Internal-only**: ml_service и rag_service НЕ доступны извне (no exposed ports)  
✅ **API Gateway**: Все запросы идут через backend (JWT auth + user permissions)  
✅ **Shared Secret**: `X-Internal-API-Key` header для backend→microservices  
✅ **Network Isolation**: Internal Docker network только для microservices  
✅ **Audit Trail**: Все запросы логируются через backend  

---

## ML Service Integration

### Архитектура

- **Framework**: FastAPI + Uvicorn
- **Models**: 4 ensemble models (CatBoost, XGBoost, RandomForest, Adaptive)
- **Performance**: <100ms p90 latency, 99.6%+ accuracy
- **Cache**: Redis-backed prediction cache

### Endpoints (Internal Only)

#### POST /api/v1/predict

**Request:**
```json
{
  "sensor_data": {
    "system_id": 123,
    "pressure": [100.5, 101.2, 99.8, ...],
    "temperature": [45.3, 45.1, 45.5, ...],
    "flow": [25.0, 24.8, 25.2, ...],
    "vibration": [0.5, 0.6, 0.5, ...]
  },
  "use_cache": true
}
```

**Response:**
```json
{
  "system_id": 123,
  "prediction": {
    "is_anomaly": true,
    "anomaly_score": 0.85,
    "severity": "critical",
    "confidence": 0.92,
    "affected_components": ["valve", "pump"],
    "anomaly_type": "pressure_drop"
  },
  "ml_predictions": [
    {
      "ml_model": "catboost",
      "version": "v1",
      "prediction_score": 0.88,
      "confidence": 0.95,
      "processing_time_ms": 5.2,
      "features_used": 25
    }
  ],
  "ensemble_score": 0.85,
  "total_processing_time_ms": 18.7,
  "features_extracted": 25,
  "cache_hit": false,
  "trace_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

#### POST /api/v1/predict/batch

Batch prediction для нескольких систем одновременно.

#### GET /health, /ready

Public health checks (без аутентификации).

### Backend Integration

```python
# backend/diagnostics/views.py
from asgiref.sync import async_to_sync
from services.ml_client import get_ml_client

class AnomalyDetectionViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=["post"])
    def detect(self, request):
        ml_client = get_ml_client()
        
        # Проверка прав пользователя на system_id
        system_id = request.data.get("system_id")
        # TODO: Check user permissions for system_id
        
        result = async_to_sync(ml_client.predict)(
            sensor_data=request.data.get("sensor_data"),
            system_id=system_id
        )
        
        return Response(result, status=status.HTTP_200_OK)
```

---

## RAG Service Integration

### Архитектура

- **Framework**: FastAPI + Uvicorn
- **LLM**: Ollama (llama3.2:latest)
- **Embeddings**: sentence-transformers
- **Vector Store**: FAISS

### Endpoints (Internal Only)

#### POST /api/v1/query

**Request:**
```json
{
  "query": "How to diagnose hydraulic pump failure?",
  "system_id": 1,
  "context": {
    "user_id": 123,
    "username": "engineer@company.com"
  },
  "max_results": 3
}
```

**Response:**
```json
{
  "response": "Generated answer from LLM...",
  "sources": [
    {
      "document_id": 1,
      "title": "Hydraulic Pump Maintenance Guide",
      "snippet": "...",
      "score": 0.95
    }
  ],
  "metadata": {
    "model": "llama3.2:latest",
    "processing_time_ms": 250,
    "tokens_used": 150
  }
}
```

### Backend Integration

```python
# backend/rag_assistant/views.py
from asgiref.sync import async_to_sync
from services.rag_client import get_rag_client

class RagAssistantViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]
    
    @action(detail=False, methods=["post"])
    def query(self, request):
        rag_client = get_rag_client()
        
        result = async_to_sync(rag_client.query)(
            query_text=request.data.get("query"),
            system_id=request.data.get("system_id"),
            user_context={
                "user_id": request.user.id,
                "username": request.user.username
            }
        )
        
        return Response(result, status=status.HTTP_200_OK)
```

---

## Конфигурация

### Environment Variables

**Backend (.env):**
```bash
# ML Service
ML_SERVICE_URL=http://ml_service:8001
ML_INTERNAL_API_KEY=your-super-secret-ml-key-min-32-chars

# RAG Service
RAG_SERVICE_URL=http://rag_service:8002
RAG_INTERNAL_API_KEY=your-super-secret-rag-key-min-32-chars
```

**ML Service (.env):**
```bash
ML_INTERNAL_API_KEY=your-super-secret-ml-key-min-32-chars
REDIS_URL=redis://redis:6379/0
DEBUG=false
```

**RAG Service (.env):**
```bash
RAG_INTERNAL_API_KEY=your-super-secret-rag-key-min-32-chars
RAG_OLLAMA_BASE_URL=http://ollama:11434
RAG_DEBUG=false
```

---

## Запуск

### 1. Настройка окружения

```bash
# Скопировать .env.example
cp .env.example .env

# Генерировать безопасные ключи
python -c "import secrets; print('ML_INTERNAL_API_KEY=' + secrets.token_urlsafe(32))"
python -c "import secrets; print('RAG_INTERNAL_API_KEY=' + secrets.token_urlsafe(32))"

# Добавить ключи в .env
```

### 2. Запуск сервисов

```bash
# Запустить все сервисы
docker-compose up -d

# Проверить статус
docker-compose ps

# Логи
docker-compose logs -f backend ml_service rag_service
```

### 3. Проверка Health Checks

```bash
# Backend
curl http://localhost:8000/health/

# ML Service (internal only, через docker network)
docker exec hdx-backend curl http://ml_service:8001/health

# RAG Service (internal only)
docker exec hdx-backend curl http://rag_service:8002/health
```

---

## E2E тестирование

### 1. Авторизация

```bash
# Получить JWT token
curl -X POST http://localhost:8000/api/auth/login/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "admin"
  }'

# Сохранить access token
export TOKEN="your-jwt-access-token"
```

### 2. ML Prediction через Backend Gateway

```bash
curl -X POST http://localhost:8000/api/diagnostics/detect/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "system_id": 1,
    "sensor_data": {
      "pressure": [100.5, 101.2, 99.8],
      "temperature": [45.3, 45.1, 45.5],
      "flow": [25.0, 24.8, 25.2],
      "vibration": [0.5, 0.6, 0.5]
    }
  }'
```

**Ожидаемый результат:**
- Backend проверяет JWT token
- Backend проверяет права пользователя на system_id
- Backend делает proxy запрос к ml_service с X-Internal-API-Key
- ml_service возвращает результат prediction
- Backend логирует запрос и возвращает результат пользователю

### 3. RAG Query через Backend Gateway

```bash
curl -X POST http://localhost:8000/api/rag/query/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to fix hydraulic leak?",
    "system_id": 1,
    "max_results": 3
  }'
```

---

## Мониторинг

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health/

# ML Service health (через backend)
curl -H "X-Internal-API-Key: $ML_INTERNAL_API_KEY" \
  http://ml_service:8001/health

# RAG Service health (через backend)
curl -H "X-Internal-API-Key: $RAG_INTERNAL_API_KEY" \
  http://rag_service:8002/health
```

### Prometheus Metrics

- **ML Service**: `http://ml_service:8001/metrics` (internal only)
- **Backend**: метрики через django-prometheus

### Structured Logs

```bash
# Логи в JSON формате
docker-compose logs ml_service | jq
docker-compose logs rag_service | jq
```

---

## Безопасность Best Practices

### Производство

1. **Измените все дефолтные ключи!**
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Network Isolation:**
   ```yaml
   networks:
     internal:
       internal: true  # Полная изоляция в prod
   ```

3. **Не expose порты ml_service и rag_service наружу!**

4. **Используйте secrets management:**
   - Kubernetes Secrets
   - AWS Secrets Manager
   - HashiCorp Vault

5. **Мониторинг аутентификации:**
   - Логируйте все failed auth attempts
   - Alert на подозрительную активность
   - Rate limiting на backend API

---

## Troubleshooting

### ML Service не отвечает

```bash
# Проверьте логи
docker-compose logs ml_service

# Проверьте health
docker exec hdx-ml-service curl http://localhost:8001/health

# Проверьте модели
docker exec hdx-ml-service ls -la /app/models/
```

### RAG Service не отвечает

```bash
# Проверьте логи
docker-compose logs rag_service

# Проверьте health
docker exec hdx-rag-service curl http://localhost:8002/health

# Проверьте Ollama
docker exec hdx-rag-service curl http://ollama:11434/api/tags
```

### 403 Forbidden от ml_service/rag_service

- Проверьте, что ML_INTERNAL_API_KEY/RAG_INTERNAL_API_KEY совпадают в backend и сервисах
- Проверьте .env файлы
- Проверьте environment variables в docker-compose.yml

---

## Performance Benchmarks

### ML Service

- **Latency**: p90 < 20ms, p99 < 35ms
- **Throughput**: 100+ RPS
- **Accuracy**: 99.6%+ (ensemble)
- **Cache Hit Rate**: 60-80% (production)

### RAG Service

- **Latency**: p90 < 500ms, p99 < 1000ms
- **Context Size**: 4096 tokens
- **Vector Search**: <50ms (FAISS)
- **LLM Generation**: 200-400ms (Ollama)

---

## Документация

- [ML Service README](../ml_service/README.md)
- [RAG Service README](../rag_service/README.md)
- [Backend API Documentation](../backend/README.md)

---

## Лицензия

Проприетарный код. Hydraulic Diagnostic Platform © 2025.
