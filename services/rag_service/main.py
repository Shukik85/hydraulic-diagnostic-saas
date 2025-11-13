# services/rag_service/main.py
"""
RAG Service FastAPI application с DeepSeek-R1.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from model_loader import get_model
from gnn_interpreter import get_interpreter

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Lifespan для загрузки модели при старте
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    logger.info("Loading DeepSeek-R1 model...")
    try:
        get_model()  # Загружаем модель
        get_interpreter()  # Инициализируем interpreter
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG Service",
    description="Reasoning AI for GNN interpretation with DeepSeek-R1",
    version="1.0.0",
    lifespan=lifespan
)


# === Request/Response Models ===

class DiagnosisInterpretRequest(BaseModel):
    """Request для интерпретации GNN diagnosis."""
    gnn_result: Dict = Field(..., description="GNN Service output")
    equipment_context: Dict = Field(..., description="Equipment metadata")
    historical_context: Optional[list] = Field(None, description="Previous diagnoses")


class DiagnosisInterpretResponse(BaseModel):
    """Response с интерпретацией."""
    summary: str = Field(..., description="Human-readable summary")
    reasoning: str = Field(..., description="Step-by-step reasoning process")
    analysis: str = Field(..., description="Detailed analysis")
    recommendations: list[str] = Field(..., description="Prioritized recommendations")
    prognosis: str = Field(..., description="Future prognosis")
    timestamp: str
    model: str
    gnn_request_id: Optional[str] = None


class AnomalyExplanationRequest(BaseModel):
    """Request для объяснения аномалии."""
    anomaly_type: str = Field(..., description="Anomaly type")
    context: Dict = Field(..., description="Additional context")


class GenerateRequest(BaseModel):
    """Generic generation request."""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(2048, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


# === Endpoints ===

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "DeepSeek-R1-Distill-32B",
        "version": "1.0.0"
    }


@app.get("/ready")
async def readiness_check():
    """Readiness check - model loaded."""
    try:
        model = get_model()
        return {"status": "ready", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")


@app.post("/interpret/diagnosis", response_model=DiagnosisInterpretResponse)
async def interpret_diagnosis(request: DiagnosisInterpretRequest):
    """
    Интерпретация GNN diagnosis results.
    
    Возвращает понятное объяснение с reasoning steps.
    """
    try:
        interpreter = get_interpreter()
        
        result = interpreter.interpret_diagnosis(
            gnn_result=request.gnn_result,
            equipment_context=request.equipment_context,
            historical_context=request.historical_context
        )
        
        return DiagnosisInterpretResponse(**result)
        
    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/anomaly")
async def explain_anomaly(request: AnomalyExplanationRequest):
    """
    Детальное объяснение конкретной аномалии.
    """
    try:
        interpreter = get_interpreter()
        
        explanation = interpreter.explain_anomaly(
            anomaly_type=request.anomaly_type,
            context=request.context
        )
        
        return {"explanation": explanation}
        
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generic text generation endpoint.
    """
    try:
        model = get_model()
        
        response = model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)}
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8004"))
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
