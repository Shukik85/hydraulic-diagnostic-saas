"""RAG query endpoints with DeepSeek-R1."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.services.auth import verify_internal_api_key

router = APIRouter()

class QueryRequest(BaseModel):
    query: str
    system_id: int
    max_results: int = 3

class QueryResponse(BaseModel):
    response: str
    sources: list[dict]
    metadata: dict

@router.post("/query", response_model=QueryResponse, dependencies=[Depends(verify_internal_api_key)])
async def query_rag(request: QueryRequest):
    """Query RAG system with DeepSeek-R1 (requires internal API key)."""
    # TODO: Implement full RAG pipeline
    return QueryResponse(
        response="RAG response placeholder (DeepSeek-R1 integration pending)",
        sources=[],
        metadata={"model": settings.llm_model, "status": "pending_implementation"}
    )
