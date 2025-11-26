"""FastAPI main application.

Production-ready inference API:
- Async endpoints
- Error handling
- CORS
- Logging
- Health checks
- Model versioning
- Request ID tracking

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.middleware import RequestIDMiddleware
from src.inference import InferenceEngine, InferenceConfig, ModelManager
from src.data import FeatureConfig
from src.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    GraphTopology,
    HealthCheckResponse,
    ModelInfo,
    ModelVersion,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global inference engine
engine: InferenceEngine | None = None
topology: GraphTopology | None = None  # Load from config/database
model_manager: ModelManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    global engine, topology, model_manager
    
    logger.info("Initializing GNN Inference Service...", extra={"request_id": "startup"})
    
    # Initialize model manager
    model_manager = ModelManager()
    
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
    
    logger.info("GNN Inference Service started successfully", extra={"request_id": "startup"})
    
    yield
    
    # Shutdown
    logger.info("Shutting down GNN Inference Service...", extra={"request_id": "shutdown"})
    # Cleanup if needed


# Create FastAPI app
app = FastAPI(
    title="GNN Inference Service",
    description="Production-ready Graph Neural Network inference for hydraulic diagnostics",
    version="2.0.0",
    lifespan=lifespan
)

# Add middlewares (order matters!)
app.add_middleware(RequestIDMiddleware)  # Request ID tracking

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],  # Expose request ID to clients
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"[{request_id}] Unhandled exception: {exc}",
        exc_info=True,
        extra={"request_id": request_id}
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred",
            "request_id": request_id
        }
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(request: Request):
    """Health check endpoint.
    
    Returns:
        status: Service health status
    
    Headers:
        X-Request-ID: Request correlation ID
    
    Examples:
        GET /health
        
        Response:
        {
            "status": "healthy",
            "version": "2.0.0",
            "model_loaded": true
        }
        
        Headers:
        X-Request-ID: abc-123-def-456
    """
    global engine
    
    request_id = getattr(request.state, "request_id", "unknown")
    logger.debug(f"[{request_id}] Health check", extra={"request_id": request_id})
    
    return HealthCheckResponse(
        status="healthy" if engine is not None else "unhealthy",
        version="2.0.0",
        model_loaded=engine is not None
    )


# Stats endpoint
@app.get("/stats")
async def get_stats(request: Request):
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
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    logger.debug(f"[{request_id}] Getting stats", extra={"request_id": request_id})
    
    stats = engine.get_stats()
    stats["request_id"] = request_id
    
    return stats


# Model versioning endpoints
@app.get("/models/versions", response_model=list[ModelVersion])
async def list_model_versions(request: Request):
    """List all available model versions.
    
    Returns:
        versions: List of model versions
    
    Examples:
        GET /models/versions
        
        Response:
        [
            {
                "version": "2.0.0",
                "path": "models/v2.0.0.ckpt",
                "size_mb": 45.3,
                "num_parameters": 2500000,
                "architecture": "GATv2-ARMA-LSTM",
                "is_current": true
            },
            ...
        ]
    """
    global model_manager, engine
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    logger.debug(f"[{request_id}] Listing model versions", extra={"request_id": request_id})
    
    # Scan models directory
    models_dir = Path("models/checkpoints")
    if not models_dir.exists():
        return []
    
    versions = []
    current_path = engine.config.model_path if engine else None
    
    for ckpt_path in models_dir.glob("*.ckpt"):
        # Extract version from filename
        version = ckpt_path.stem
        
        # Get file size
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
        
        # Get model info if loaded
        info = model_manager.get_model_info(str(ckpt_path))
        num_params = info["num_parameters"] if info else 0
        
        versions.append(
            ModelVersion(
                version=version,
                path=str(ckpt_path),
                size_mb=size_mb,
                num_parameters=num_params,
                architecture="GATv2-ARMA-LSTM",
                is_current=(str(ckpt_path) == current_path),
                created_at=datetime.fromtimestamp(ckpt_path.stat().st_mtime)
            )
        )
    
    return sorted(versions, key=lambda v: v.created_at, reverse=True)


@app.get("/models/current", response_model=ModelInfo | None)
async def get_current_model(request: Request):
    """Get currently active model info.
    
    Returns:
        info: Current model information
    
    Examples:
        GET /models/current
        
        Response:
        {
            "path": "models/checkpoints/best.ckpt",
            "version": "2.0.0",
            "device": "cuda:0",
            "num_parameters": 2500000,
            "size_mb": 45.3,
            "loaded": true,
            "loaded_at": "2025-11-26T20:00:00Z",
            "compiled": true
        }
    """
    global model_manager, engine
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    if engine is None or model_manager is None:
        return None
    
    logger.debug(f"[{request_id}] Getting current model", extra={"request_id": request_id})
    
    model_path = engine.config.model_path
    info = model_manager.get_model_info(model_path)
    
    if info is None:
        return None
    
    # Get file size
    path_obj = Path(model_path)
    size_mb = path_obj.stat().st_size / (1024 * 1024) if path_obj.exists() else 0
    
    return ModelInfo(
        path=model_path,
        version=path_obj.stem,
        device=info["device"],
        num_parameters=info["num_parameters"],
        size_mb=size_mb,
        loaded=True,
        loaded_at=datetime.now(),  # TODO: Track actual load time
        compiled=info.get("compiled", False)
    )


# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request_data: PredictionRequest, request: Request):
    """Single equipment prediction.
    
    Args:
        request_data: Prediction request
        request: FastAPI request (for request ID)
    
    Returns:
        response: Prediction response
    
    Examples:
        POST /predict
        X-Request-ID: my-correlation-id
        
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
            "anomaly": {...},
            "inference_time_ms": 45.3
        }
        
        Headers:
        X-Request-ID: my-correlation-id
    """
    global engine, topology
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if topology is None:
        raise HTTPException(status_code=500, detail="Topology not configured")
    
    try:
        logger.info(
            f"[{request_id}] Prediction request for {request_data.equipment_id}",
            extra={"request_id": request_id, "equipment_id": request_data.equipment_id}
        )
        
        response = await engine.predict(
            request=request_data,
            topology=topology
        )
        
        logger.info(
            f"[{request_id}] Prediction complete for {request_data.equipment_id} - "
            f"health={response.health.score:.2f}",
            extra={
                "request_id": request_id,
                "equipment_id": request_data.equipment_id,
                "health_score": response.health.score
            }
        )
        
        return response
    
    except Exception as e:
        logger.error(
            f"[{request_id}] Prediction failed: {e}",
            exc_info=True,
            extra={"request_id": request_id, "equipment_id": request_data.equipment_id}
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request_data: BatchPredictionRequest, request: Request):
    """Batch equipment predictions.
    
    Args:
        request_data: Batch prediction request
        request: FastAPI request (for request ID)
    
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
            "predictions": [...],
            "total_count": 10,
            "total_time_ms": 234.5
        }
    """
    global engine, topology
    
    request_id = getattr(request.state, "request_id", "unknown")
    
    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if topology is None:
        raise HTTPException(status_code=500, detail="Topology not configured")
    
    try:
        import time
        
        logger.info(
            f"[{request_id}] Batch prediction request for {len(request_data.requests)} equipment",
            extra={"request_id": request_id, "batch_size": len(request_data.requests)}
        )
        
        start_time = time.time()
        
        predictions = await engine.predict_batch(
            requests=request_data.requests,
            topology=topology
        )
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(
            f"[{request_id}] Batch prediction complete - {len(predictions)} results in {total_time:.2f}ms",
            extra={
                "request_id": request_id,
                "batch_size": len(predictions),
                "total_time_ms": total_time
            }
        )
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            total_time_ms=total_time
        )
    
    except Exception as e:
        logger.error(
            f"[{request_id}] Batch prediction failed: {e}",
            exc_info=True,
            extra={"request_id": request_id}
        )
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
