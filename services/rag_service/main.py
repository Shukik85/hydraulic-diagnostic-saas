# services/rag_service/main.py
"""
RAG Service - Complete implementation with admin endpoints.
"""
import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from model_loader import get_model
from gnn_interpreter import get_interpreter
from admin_endpoints import router as admin_router
from openapi_config import custom_openapi

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    logger.info("Loading DeepSeek-R1 model...")
    try:
        get_model()
        get_interpreter()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG Service API",
    version="1.0.0",
    description="Reasoning AI for GNN interpretation with DeepSeek-R1",
    openapi_version="3.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include admin router
app.include_router(admin_router)

# Apply custom OpenAPI
app.openapi = lambda: custom_openapi(app)


# === Request/Response Models ===

class DiagnosisInterpretRequest(BaseModel):
    """Request для интерпретации GNN diagnosis."""
    gnn_result: Dict = Field(..., description="GNN Service output")
    equipment_context: Dict = Field(..., description="Equipment metadata")
    historical_context: Optional[List[Dict]] = Field(None, description="Previous diagnoses")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gnn_result": {
                    "overall_health_score": 0.65,
                    "component_health": [
                        {
                            "component_id": "pump_001",
                            "health_score": 0.65,
                            "degradation_rate": 0.08
                        }
                    ],
                    "anomalies": [
                        {
                            "anomaly_type": "pressure_drop",
                            "severity": "high",
                            "confidence": 0.85
                        }
                    ]
                },
                "equipment_context": {
                    "equipment_id": "exc_001",
                    "equipment_type": "excavator",
                    "model": "CAT-320D",
                    "operating_hours": 8500
                }
            }
        }


class DiagnosisInterpretResponse(BaseModel):
    """Response с интерпретацией."""
    summary: str = Field(..., description="Human-readable summary")
    reasoning: str = Field(..., description="Step-by-step reasoning")
    analysis: str = Field(..., description="Detailed analysis")
    recommendations: List[str] = Field(..., description="Prioritized recommendations")
    prognosis: str = Field(..., description="Future prognosis")
    timestamp: str
    model: str
    gnn_request_id: Optional[str] = None


class AnomalyExplanationRequest(BaseModel):
    """Request для объяснения аномалии."""
    anomaly_type: str = Field(..., description="Type of anomaly")
    context: Dict = Field(..., description="Context data")


class GenerateRequest(BaseModel):
    """Generic generation request."""
    prompt: str = Field(..., description="Input prompt")
    max_tokens: int = Field(2048, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


# === Monitoring Endpoints ===

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Service health check.
    
    Returns service status and model availability.
    """
    try:
        model = get_model()
        return {
            "service": "rag-service",
            "status": "healthy",
            "model": "DeepSeek-R1-Distill-32B",
            "model_loaded": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "service": "rag-service",
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/ready", tags=["Monitoring"])
async def readiness_check():
    """Readiness check - model loaded."""
    try:
        model = get_model()
        return {"status": "ready", "model_loaded": True}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Not ready: {e}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# === Inference Endpoints ===

@app.post("/interpret/diagnosis", response_model=DiagnosisInterpretResponse, tags=["Interpretation"])
async def interpret_diagnosis(request: DiagnosisInterpretRequest):
    """
    Интерпретация GNN diagnosis results.
    
    Преобразует технические GNN outputs в понятные рекомендации.
    
    **Process**:
    1. Analyze GNN results
    2. Apply reasoning (DeepSeek-R1)
    3. Generate human-readable summary
    4. Prioritize recommendations
    5. Predict future issues
    
    **Latency**: 2-3 seconds typical
    """
    try:
        interpreter = get_interpreter()
        
        result = interpreter.interpret_diagnosis(
            gnn_result=request.gnn_result,
            equipment_context=request.equipment_context,
            historical_context=request.historical_context
        )
        
        logger.info(f"Interpreted diagnosis for {request.equipment_context.get('equipment_id')}")
        
        return DiagnosisInterpretResponse(**result)
        
    except Exception as e:
        logger.error(f"Interpretation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Interpretation failed: {str(e)}")


@app.post("/explain/anomaly", tags=["Explanation"])
async def explain_anomaly(request: AnomalyExplanationRequest):
    """
    Детальное объяснение конкретной аномалии.
    
    **Use case**: User clicks on anomaly, wants detailed explanation.
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


@app.post("/generate", tags=["Generation"])
async def generate(request: GenerateRequest):
    """
    Generic text generation endpoint.
    
    **Warning**: This is a powerful endpoint. Use with caution.
    **Access**: Consider restricting in production.
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


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "service": "RAG Service",
        "version": "1.0.0",
        "model": "DeepSeek-R1-Distill-32B",
        "status": "operational",
        "docs": "/docs"
    }


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
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
