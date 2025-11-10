# ============================================================================
# RAG Service README (минимум для bootstrap)
# ============================================================================

# Hydraulic Diagnostic Platform — RAG Service

**RAG Service** — Retrieval-Augmented Generation сервис для диагностики
гидросистем на базе DeepSeek-R1, FAISS/Qdrant, Ollama и LangChain.

### Features
- RAG pipeline: question → search context → LLM answer (DeepSeek-R1)
- FAISS (dev), Qdrant (prod-ready) для поиска релевантных docs
- Multilingual embeddings (RU/EN)
- Prometheus metрики, health check endpoint
- Explainability QA (интеграция attention GNN + историка)
- Безопасность (internal API key)
- Готовность к cloud/k8s scaling (Docker, deploy scripts)

### Быстрый старт

```bash
cd rag_service
cp .env.example .env
# (base image уже есть)  
docker build -t hdx-rag-service:latest .
docker run --rm -p 8002:8002 \
  -v $(pwd)/data:/data \
  --env-file .env \
  hdx-rag-service:latest
```

### Тестовый запрос

```bash
curl -X POST http://localhost:8002/rag/query \
  -H "X-Internal-API-Key: changeme-rag-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Почему pump влияет на boom?", "language": "ru"}'
```
