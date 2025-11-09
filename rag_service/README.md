# RAG Service — Internal Microservice

**Internal-only RAG (Retrieval-Augmented Generation) service** для Hydraulic Diagnostic Platform.

## Обзор

- **Архитектура:** FastAPI async microservice
- **Доступ:** Только через internal Docker network (backend → rag_service)
- **Аутентификация:** Shared secret (`X-Internal-API-Key` header)
- **Назначение:** Обработка RAG-запросов от backend, embeddings, vector search, LLM generation

## Компоненты

```
rag_service/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── config.py         # Pydantic settings
│   ├── auth.py           # Internal API key auth
│   └── routes/
│       ├── health.py     # Health checks
│       └── rag.py        # RAG query endpoints
├── requirements.txt      # Dependencies
└── README.md
```

## Технологии

- **Framework:** FastAPI + Uvicorn
- **LLM:** Ollama (llama3.2:latest)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2
- **Vector Store:** FAISS
- **Logging:** structlog (JSON structured logs)

## Endpoints

### Public (No Auth)

- `GET /health` — Health check
- `GET /health/ready` — Readiness probe
- `GET /health/live` — Liveness probe

### Internal (Requires X-Internal-API-Key)

- `POST /api/v1/query` — Process RAG query

#### Example Request:

```bash
curl -X POST http://rag_service:8002/api/v1/query \
  -H "X-Internal-API-Key: your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How to diagnose hydraulic pump failure?",
    "system_id": 1,
    "context": {"user_id": 123},
    "max_results": 3
  }'
```

#### Response:

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

## Конфигурация

### Environment Variables (via `.env` or Docker Compose)

```bash
# Security
RAG_INTERNAL_API_KEY=your-super-secret-key-here

# LLM Configuration
RAG_LLM_MODEL=llama3.2:latest
RAG_OLLAMA_BASE_URL=http://ollama:11434

# Embeddings
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_VECTOR_STORE_PATH=/data/faiss_index

# Performance
RAG_MAX_CONTEXT_LENGTH=4096
RAG_CHUNK_SIZE=512
RAG_CHUNK_OVERLAP=50

# Logging
RAG_LOG_LEVEL=INFO
RAG_DEBUG=false
```

## Запуск

### Development (local)

```bash
cd rag_service
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8002
```

### Production (Docker)

```bash
docker-compose up rag_service
```

## Интеграция с Backend

Backend использует async HTTP client (`httpx`) для вызова rag_service:

```python
# backend/services/rag_client.py
from services.rag_client import get_rag_client

rag_client = get_rag_client()
response = await rag_client.query(
    query_text="How to fix hydraulic leak?",
    system_id=1,
    user_context={"user_id": 123}
)
```

## Безопасность

✅ **Internal-only:** Порты НЕ exposed наружу (только internal Docker network)  
✅ **API Key:** Все запросы требуют `X-Internal-API-Key` header  
✅ **No user access:** Пользователи НЕ могут вызывать rag_service напрямую  
✅ **Audit logging:** Все запросы логируются через backend  

## Мониторинг

- **Health checks:** `/health`, `/health/ready`, `/health/live`
- **Structured logging:** JSON logs via structlog
- **Metrics:** TODO: Prometheus metrics endpoint

## TODO

- [ ] Реализовать RAG pipeline (vector search + LLM generation)
- [ ] Интеграция с FAISS vector store
- [ ] Загрузка Ollama models
- [ ] Prometheus metrics endpoint
- [ ] Unit/integration tests
- [ ] Performance benchmarks

## Лицензия

Проприетарный код. Внутренний микросервис Hydraulic Diagnostic Platform.
