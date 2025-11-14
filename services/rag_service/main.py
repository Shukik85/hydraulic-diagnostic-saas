# services/rag_service/main.py
"""
RAG Service - Production-ready v2.0

Features:
- Rate limiting (GPU protection)
- Structured logging (JSON for ELK/Grafana)
- Config-based settings
- Request ID tracing
"""
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn
import structlog

from model_loader import get_model
from gnn_interpreter import get_interpreter
from admin_endpoints import router as admin_router
from openapi_config import custom_openapi
from config import config

# Structured Logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if config.LOG_FORMAT == "json" 
            else structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Rate Limiter
limiter = Limiter(
    key_func=get_remote_address,
    enabled=config.RATE_LIMIT_ENABLED
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown."""
    logger.info(
        "service_starting",
        service=config.SERVICE_NAME,
        version=config.SERVICE_VERSION,
        environment=config.ENVIRONMENT
    )
    
    try:
        get_model()
        get_interpreter()
        logger.info("model_loaded", model=config.MODEL_NAME)
    except Exception as e:
        logger.error("model_load_failed", error=str(e))
        raise
    
    yield
    logger.info("service_shutting_down")


app = FastAPI(
    title=config.SERVICE_NAME,
    version=config.SERVICE_VERSION,
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.openapi = lambda: custom_openapi(app)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Request ID tracing."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method
    )
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    structlog.contextvars.clear_contextvars()
    return response


app.include_router(admin_router)


# Models
class DiagnosisInterpretRequest(BaseModel):
    gnn_result: Dict
    equipment_context: Dict
    historical_context: Optional[List[Dict]] = None


class DiagnosisInterpretResponse(BaseModel):
    summary: str
    reasoning: str
    analysis: str
    recommendations: List[str]
    prognosis: str
    timestamp: str
    model: str
    gnn_request_id: Optional[str] = None


class AnomalyExplanationRequest(BaseModel):
    anomaly_type: str
    context: Dict


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


# Endpoints
@app.get("/")
async def root():
    return {
        "service": config.SERVICE_NAME,
        "version": config.SERVICE_VERSION,
        "status": "operational",
        "environment": config.ENVIRONMENT
    }


@app.get("/health")
async def health_check():
    try:
        get_model()
        model_healthy = True
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        model_healthy = False
    
    return {
        "status": "healthy" if model_healthy else "degraded",
        "model_loaded": model_healthy
    }


@app.post("/interpret/diagnosis", response_model=DiagnosisInterpretResponse)
@limiter.limit(config.RATE_LIMIT_DIAGNOSIS)
async def interpret_diagnosis(
    request_data: DiagnosisInterpretRequest,
    request: Request
):
    """GNN interpretation с reasoning. Rate: {config.RATE_LIMIT_DIAGNOSIS}"""
    equipment_id = request_data.equipment_context.get("equipment_id", "unknown")
    
    logger.info(
        "diagnosis_started",
        equipment_id=equipment_id,
        has_history=request_data.historical_context is not None
    )
    
    try:
        interpreter = get_interpreter()
        result = interpreter.interpret_diagnosis(
            gnn_result=request_data.gnn_result,
            equipment_context=request_data.equipment_context,
            historical_context=request_data.historical_context
        )
        
        logger.info("diagnosis_completed", equipment_id=equipment_id)
        return DiagnosisInterpretResponse(**result)
        
    except Exception as e:
        logger.error("diagnosis_failed", equipment_id=equipment_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/anomaly")
@limiter.limit(config.RATE_LIMIT_EXPLAIN)
async def explain_anomaly(
    request_data: AnomalyExplanationRequest,
    request: Request
):
    """Anomaly explanation. Rate: {config.RATE_LIMIT_EXPLAIN}"""
    logger.info("anomaly_explain_started", anomaly_type=request_data.anomaly_type)
    
    try:
        interpreter = get_interpreter()
        explanation = interpreter.explain_anomaly(
            anomaly_type=request_data.anomaly_type,
            context=request_data.context
        )
        
        logger.info("anomaly_explain_completed")
        return {"explanation": explanation}
        
    except Exception as e:
        logger.error("anomaly_explain_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
@limiter.limit(config.RATE_LIMIT_GENERATE)
async def generate(
    request_data: GenerateRequest,
    request: Request
):
    """Generic generation. Rate: {config.RATE_LIMIT_GENERATE}"""
    logger.info(
        "generate_started",
        prompt_length=len(request_data.prompt),
        max_tokens=request_data.max_tokens
    )
    
    try:
        model = get_model()
        response = model.generate(
            prompt=request_data.prompt,
            max_tokens=request_data.max_tokens,
            temperature=request_data.temperature
        )
        
        logger.info("generate_completed", response_length=len(response))
        return {"response": response}
        
    except Exception as e:
        logger.error("generate_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "request_id": getattr(request.state, "request_id", None)}
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    )
