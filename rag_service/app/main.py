"""RAG Service API (FastAPI).

- POST /rag/query — основное взаимодействие (chain-of-thought QA)
- POST /rag/attention_explain — запрос attention-weight анализа от GNN/DB
- POST /rag/history — временные срезы истории диагностики
- GET /health — health check endpoint
- GET /metrics — Prometheus metrics

"""

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest

from app.config import get_settings
from app.auth import verify_api_key
from app.schemas import (
    RAGQueryRequest,
    RAGQueryResponse,
    RAGAttentionExplainRequest,
    RAGAttentionExplainResponse,
    RAGHistoryRequest,
    RAGHistoryResponse,
    HealthResponse,
)
from app.services.rag_pipeline import DeepSeekRAGPipeline

app = FastAPI(title="RAG Service",
              description="Retrieval-Augmented Generation for hydraulic diagnostics",
              version="0.1.0")
settings = get_settings()

# Prometheus metrics
rag_query_counter = Counter('rag_query_total', 'Total RAG queries')
rag_query_duration = Histogram('rag_query_duration_seconds', 'RAG query latency in seconds')

pipeline = DeepSeekRAGPipeline()

@app.get("/health", response_model=HealthResponse)
async def health():
    # Проверка готовности FAISS, Ollama, embeddings, API key
    ok = pipeline.ready()
    return HealthResponse(status="healthy" if ok else "unhealthy",
                         faiss=pipeline.faiss_ready(),
                         ollama=pipeline.ollama_ready(),
                         model_loaded=pipeline.model_loaded())

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.post("/rag/query", response_model=RAGQueryResponse)
@rag_query_duration.time()
async def rag_query(req: RAGQueryRequest, api_key: str = Depends(verify_api_key)):
    rag_query_counter.inc()
    try:
        response = await pipeline.query(
            question=req.question,
            context=req.context,
            equipment_id=req.equipment_id,
            language=req.language,
        )
        return RAGQueryResponse(**response)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/rag/attention_explain", response_model=RAGAttentionExplainResponse)
async def rag_attention_explain(req: RAGAttentionExplainRequest, api_key: str = Depends(verify_api_key)):
    try:
        result = await pipeline.attention_explain(
            attention_weights=req.attention_weights,
            equipment_id=req.equipment_id,
            gnn_reasoning=req.gnn_reasoning,
            language=req.language,
        )
        return RAGAttentionExplainResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/rag/history", response_model=RAGHistoryResponse)
async def rag_history(req: RAGHistoryRequest, api_key: str = Depends(verify_api_key)):
    try:
        result = await pipeline.history(
            equipment_id=req.equipment_id,
            since=req.since,
            until=req.until,
            max_docs=req.max_docs,
            language=req.language,
        )
        return RAGHistoryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
