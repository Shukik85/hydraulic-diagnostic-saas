# FAISS Index Build & Import Admin API для RAG Service
from fastapi import APIRouter, Depends, HTTPException, status
from app.auth import verify_api_key
from app.config import get_settings
from app.services.faiss_indexer import FAISSIndexer
from typing import List, Dict, Any
import json
settings = get_settings()
router = APIRouter(prefix='/rag/admin', tags=['RAG Admin'])
indexer = FAISSIndexer(settings.faiss_index_path, settings.embedding_model)
@router.post('/rebuild')
async def rebuild_index(docs: List[Dict[str, Any]], api_key: str = Depends(verify_api_key)):
    # docs: список словарей {text: str, meta: dict}
    try:
        indexer.build_index_from_docs(docs)
        return {"status": "ok", "count": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
@router.post('/add_docs')
async def add_docs(docs: List[Dict[str, Any]], api_key: str = Depends(verify_api_key)):
    try:
        indexer.add_docs(docs)
        return {"status": "ok", "added": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
@router.get('/info')
async def index_info(api_key: str = Depends(verify_api_key)):
    count = indexer.load_index()
    return {"status": "ok", "count": count}
