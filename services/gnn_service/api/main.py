"""FastAPI main application.

Production-ready inference API:
- Async endpoints
- Error handling
- CORS
- Logging
- Health checks

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from src.inference import InferenceEngine, InferenceConfig
from src.data import FeatureConfig
from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    GraphTopology,
    HealthCheckResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global inference engine
engine: InferenceEngine | None = None
topology: GraphTopology | None = None  # Load from config/database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    global engine, topology
    
    logger.info("Initializing GNN Inference Service...")
    
    # Initialize inference engine
    inference_config = InferenceConfig(
        model_path="models/checkpoints/best.ckpt",  # TODO: Load from env
        device="auto",
        batch_size=32,
        use_dynamic_batching=True
    )
    
    feature_config = FeatureConfig(
        use_statistical=True,
        use_frequency=True,
        use_temporal=True,
        use_hydraulic=True
    )
    
    engine = InferenceEngine(
        config=inference_config,
        feature_config=feature_config
    )
    
    # TODO: Load topology from database/config
    # topology = load_topology(...)
    
    logger.info("GNN Inference Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GNN Inference Service...")
    # Cleanup if needed


# Create FastAPI app
app = FastAPI(
    title="GNN Inference Service",
    description="Production-ready Graph Neural Network inference for hydraulic diagnostics",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred"
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint.
    
    Returns:
        status: Service health status
    
    Examples:
        GET /health
        
        Response:
        {
            "status": "healthy",
            "version": "2.0.0",
            "model_loaded": true
        }
    """
    global engine
    
    return HealthCheckResponse(
        status="healthy" if engine is not None else "unhealthy",
        version="2.0.0",
        model_loaded=engine is not None
    )


# Stats endpoint
@app.get("/stats")
async def get_stats():
    """Get service statistics.
    
    Returns:
        stats: Service statistics
    
    Examples:
        GET /stats
        
        Response:
        {
            "model_path": "models/best.ckpt",
            "device": "cuda",
            "batch_size": 32,
            ...
        }
    """
    global engine
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return engine.get_stats()


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Single equipment prediction.
    
    Args:
        request: Prediction request
    
    Returns:
        response: Prediction response
    
    Examples:
        POST /predict
        
        Request:
        {
            "equipment_id": "exc_001",
            "sensor_data": {...}
        }
        
        Response:
        {
            "equipment_id": "exc_001",
            "health": {"score": 0.85},
            "degradation": {"rate": 0.12},
            "anomaly": {
                "predictions": {
                    "pressure_drop": 0.05,
                    "overheating": 0.03,
                    ...
                }
            },
            "inference_time_ms": 45.3
        }
    """
    global engine, topology
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if topology is None:
        raise HTTPException(status_code=500, detail="Topology not configured")
    
    try:
        response = await engine.predict(
            request=request,
            topology=topology
        )
        return response
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Batch equipment predictions.
    
    Args:
        request: Batch prediction request
    
    Returns:
        response: Batch prediction response
    
    Examples:
        POST /predict/batch
        
        Request:
        {
            "requests": [
                {"equipment_id": "exc_001", "sensor_data": {...}},
                {"equipment_id": "exc_002", "sensor_data": {...}},
                ...
            ]
        }
        
        Response:
        {
            "predictions": [
                {"equipment_id": "exc_001", "health": {...}, ...},
                {"equipment_id": "exc_002", "health": {...}, ...},
                ...
            ],
            "total_count": 10,
            "total_time_ms": 234.5
        }
    """
    global engine, topology
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if topology is None:
        raise HTTPException(status_code=500, detail="Topology not configured")
    
    try:
        import time
        start_time = time.time()
        
        predictions = await engine.predict_batch(
            requests=request.requests,
            topology=topology
        )
        
        total_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            total_time_ms=total_time
        )
    
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
