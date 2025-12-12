"""FastAPI main application.

Production-ready inference API:
- Async endpoints
- Error handling
- CORS
- Logging
- Health checks (basic + detailed)
- Model versioning
- Request ID tracking

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import psutil
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware import RequestIDMiddleware
from src.data import FeatureConfig
from src.inference import InferenceConfig, InferenceEngine, ModelManager
from src.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ComponentHealth,
    ComponentStatus,
    DetailedHealthResponse,
    GraphTopology,
    HealthCheckResponse,
    ModelInfo,
    ModelVersion,
    PredictionRequest,
    PredictionResponse,
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

# Global state
engine: InferenceEngine | None = None
topology: GraphTopology | None = None
model_manager: ModelManager | None = None
start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    global engine, topology, model_manager, start_time

    start_time = time.time()

    logger.info("Initializing GNN Inference Service...", extra={"request_id": "startup"})

    # Initialize model manager
    model_manager = ModelManager()

    # Initialize inference engine
    inference_config = InferenceConfig(
        model_path="models/checkpoints/best.ckpt",
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

    logger.info("GNN Inference Service started successfully", extra={"request_id": "startup"})

    yield

    # Shutdown
    logger.info("Shutting down GNN Inference Service...", extra={"request_id": "shutdown"})


# Create FastAPI app
app = FastAPI(
    title="GNN Inference Service",
    description="Production-ready Graph Neural Network inference for hydraulic diagnostics",
    version="2.0.0",
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
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


# Health check endpoints
@app.get("/health", response_model=HealthCheckResponse)
async def health_check(request: Request):
    """Basic health check."""
    global engine

    return HealthCheckResponse(
        status="healthy" if engine is not None else "unhealthy",
        version="2.0.0",
        model_loaded=engine is not None
    )


@app.get("/health/detailed", response_model=DetailedHealthResponse)
async def detailed_health_check(request: Request):
    """Detailed health check with system metrics.
    
    Returns:
        Detailed health status including:
        - Component statuses
        - System metrics (CPU, memory, disk)
        - Model information
        - Service uptime
    
    Examples:
        GET /health/detailed
        
        Response:
        {
            "status": "healthy",
            "version": "2.0.0",
            "uptime_seconds": 3600.0,
            "components": {...},
            "system_metrics": {
                "cpu_percent": 25.3,
                "memory_percent": 45.7,
                "disk_percent": 35.2
            },
            "model_info": {...}
        }
    """
    global engine, model_manager, start_time

    # Calculate uptime
    uptime = time.time() - start_time

    # Check components
    components = {}

    # Check inference engine
    if engine is not None:
        components["inference_engine"] = ComponentHealth(
            name="inference_engine",
            status=ComponentStatus.HEALTHY,
            message="Ready for inference"
        )
    else:
        components["inference_engine"] = ComponentHealth(
            name="inference_engine",
            status=ComponentStatus.FAILED,
            message="Not initialized"
        )

    # Check model manager
    if model_manager is not None:
        components["model_manager"] = ComponentHealth(
            name="model_manager",
            status=ComponentStatus.HEALTHY,
            message="Model loaded"
        )
    else:
        components["model_manager"] = ComponentHealth(
            name="model_manager",
            status=ComponentStatus.FAILED,
            message="Not initialized"
        )

    # Get system metrics
    system_metrics = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent
    }

    # Get model info
    model_info: dict[str, bool | str | int] = {"loaded": False}

    if engine and model_manager:
        info = model_manager.get_model_info(engine.config.model_path)
        if info:
            model_info = {
                "loaded": True,
                "device": info["device"],
                "parameters": info["num_parameters"],
                "compiled": info.get("compiled", False)
            }

    # Determine overall status
    if all(c.status == ComponentStatus.HEALTHY for c in components.values()):
        overall_status = "healthy"
    elif any(c.status == ComponentStatus.FAILED for c in components.values()):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"

    return DetailedHealthResponse(
        status=overall_status,  # type: ignore
        version="2.0.0",
        uptime_seconds=uptime,
        components=components,
        system_metrics=system_metrics,
        model_info=model_info
    )


# Stats endpoint
@app.get("/stats")
async def get_stats(request: Request):
    """Get service statistics."""
    global engine

    request_id = getattr(request.state, "request_id", "unknown")

    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    stats = engine.get_stats()
    stats["request_id"] = request_id

    return stats


# Model versioning endpoints
@app.get("/models/versions", response_model=list[ModelVersion])
async def list_model_versions(request: Request):
    """List all available model versions."""
    global model_manager, engine

    if model_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    models_dir = Path("models/checkpoints")
    if not models_dir.exists():
        return []

    versions = []
    current_path = engine.config.model_path if engine else None

    for ckpt_path in models_dir.glob("*.ckpt"):
        version = ckpt_path.stem
        size_mb = ckpt_path.stat().st_size / (1024 * 1024)
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
    """Get currently active model."""
    global model_manager, engine

    if engine is None or model_manager is None:
        return None

    model_path = engine.config.model_path
    info = model_manager.get_model_info(model_path)

    if info is None:
        return None

    path_obj = Path(model_path)
    size_mb = path_obj.stat().st_size / (1024 * 1024) if path_obj.exists() else 0

    return ModelInfo(
        path=model_path,
        version=path_obj.stem,
        device=info["device"],
        num_parameters=info["num_parameters"],
        size_mb=size_mb,
        loaded=True,
        loaded_at=datetime.now(),
        compiled=info.get("compiled", False)
    )


# Prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(request_data: PredictionRequest, request: Request):
    """Single equipment prediction."""
    global engine, topology

    request_id = getattr(request.state, "request_id", "unknown")

    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if topology is None:
        raise HTTPException(status_code=500, detail="Topology not configured")

    try:
        logger.info(
            f"[{request_id}] Prediction for {request_data.equipment_id}",
            extra={"request_id": request_id, "equipment_id": request_data.equipment_id}
        )

        response = await engine.predict(request=request_data, topology=topology)
        return response

    except Exception as e:
        logger.error(f"[{request_id}] Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request_data: BatchPredictionRequest, request: Request):
    """Batch equipment predictions."""
    global engine, topology

    request_id = getattr(request.state, "request_id", "unknown")

    if engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    if topology is None:
        raise HTTPException(status_code=500, detail="Topology not configured")

    try:
        start = time.time()

        predictions = await engine.predict_batch(
            requests=request_data.requests,
            topology=topology
        )

        total_time = (time.time() - start) * 1000

        return BatchPredictionResponse(
            predictions=predictions,
            total_count=len(predictions),
            total_time_ms=total_time
        )

    except Exception as e:
        logger.error(f"[{request_id}] Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e!s}")


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
