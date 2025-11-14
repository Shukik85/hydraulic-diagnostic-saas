# services/rag_service/main.py
"""
RAG Service - Updated with admin endpoints.
"""
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model_loader import get_model
from gnn_interpreter import get_interpreter
from admin_endpoints import router as admin_router
from openapi_config import custom_openapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    logger.info("Loading DeepSeek-R1 model...")
    try:
        get_model()
        get_interpreter()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
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

# Include routers
app.include_router(admin_router)

# Apply custom OpenAPI
app.openapi = lambda: custom_openapi(app)


# === Health & Metrics ===

@app.get("/health", tags=["Monitoring"])
async def health_check():
    """Service health check."""
    return {
        "service": "rag-service",
        "status": "healthy",
        "model": "DeepSeek-R1-Distill-32B",
        "version": "1.0.0"
    }


@app.get("/ready", tags=["Monitoring"])
async def readiness_check():
    """Readiness check - model loaded."""
    try:
        model = get_model()
        return {"status": "ready", "model_loaded": True}
    except Exception as e:
        return {"status": "not_ready", "error": str(e)}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# === Existing endpoints ===
# (Keep /interpret/diagnosis, /explain/anomaly, /generate)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8004"))
    uvicorn.run(app, host="0.0.0.0", port=port)
