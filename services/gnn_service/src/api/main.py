"""Production-Ready GNN Service FastAPI Application.

Universal Temporal GNN (GAT + LSTM) for multi-label classification
of hydraulic system component states.

Entry Point: uvicorn src.api.main:app
"""

from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

try:
    from src.schemas.requests import MinimalInferenceRequest, PredictionRequest
    from src.schemas.metadata import EquipmentMetadata
    from src.inference.inference_engine import InferenceEngine, InferenceConfig
except ImportError:
    # Fallback for development
    from schemas.requests import MinimalInferenceRequest, PredictionRequest
    from schemas.metadata import EquipmentMetadata
    from inference.inference_engine import InferenceEngine, InferenceConfig


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

inference_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """FastAPI lifespan context manager.
    
    Startup: Initialize inference engine
    Shutdown: Cleanup resources
    """
    global inference_engine
    
    # Startup
    try:
        config = InferenceConfig()
        inference_engine = InferenceEngine(config)
        print("✅ Inference engine initialized")
    except Exception as e:
        print(f"⚠️  Warning: Inference engine initialization failed: {e}")
        print("   API will operate in limited mode")
    
    yield
    
    # Shutdown
    if inference_engine:
        try:
            inference_engine.cleanup()
            print("✅ Inference engine cleaned up")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")


# ============================================================================
# APP CREATION
# ============================================================================

app = FastAPI(
    title="GNN Service",
    description="Universal Temporal GNN for Hydraulic System Diagnostics",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HEALTH CHECK ENDPOINTS
# ============================================================================


@app.get(
    "/health",
    tags=["Health"],
    summary="Service health check",
    response_model=dict[str, Any],
)
async def health_check() -> dict[str, Any]:
    """Check if service is running.
    
    Returns:
        dict: Status information
    """
    return {
        "status": "healthy",
        "service": "gnn-service",
        "version": "1.0.0",
    }


@app.get(
    "/ready",
    tags=["Health"],
    summary="Service readiness check",
    response_model=dict[str, Any],
)
async def readiness_check() -> dict[str, Any]:
    """Check if service is ready for requests.
    
    Returns:
        dict: Readiness status
        
    Raises:
        HTTPException: If service not ready
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    return {
        "ready": True,
        "inference_engine": "initialized",
    }


@app.get(
    "/metrics",
    tags=["Health"],
    summary="Service metrics",
    response_model=dict[str, Any],
)
async def get_metrics() -> dict[str, Any]:
    """Get service metrics and statistics.
    
    Returns:
        dict: Current metrics
    """
    if inference_engine is None:
        return {"status": "engine_not_initialized"}
    
    try:
        stats = inference_engine.get_stats()
        return {
            "inference_engine": stats,
            "service_version": "1.0.0",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
        }


# ============================================================================
# INFERENCE ENDPOINTS
# ============================================================================


@app.post(
    "/v1/diagnose",
    tags=["Inference"],
    summary="Run GNN diagnosis",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def run_diagnosis(request: MinimalInferenceRequest) -> dict[str, Any]:
    """Run GNN-based diagnosis on hydraulic system.
    
    Args:
        request: Inference request with system readings
        
    Returns:
        dict: Diagnosis results with predictions
        
    Raises:
        HTTPException: If inference fails
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    try:
        # Run inference
        result = await inference_engine.infer(request)
        
        return {
            "status": "success",
            "equipment_id": request.equipment_id,
            "timestamp": request.timestamp.isoformat(),
            "diagnosis": result,
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@app.post(
    "/v1/predict",
    tags=["Inference"],
    summary="Get predictions",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def get_predictions(request: PredictionRequest) -> dict[str, Any]:
    """Get detailed predictions for system components.
    
    Args:
        request: Prediction request
        
    Returns:
        dict: Detailed predictions per component
        
    Raises:
        HTTPException: If prediction fails
    """
    if inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialized",
        )
    
    try:
        predictions = await inference_engine.predict(request)
        return {
            "status": "success",
            "predictions": predictions,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(
    request: Any, exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Any, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions."""
    print(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# ============================================================================
# INFO ENDPOINT
# ============================================================================


@app.get(
    "/",
    tags=["Info"],
    summary="Service information",
    response_model=dict[str, Any],
)
async def get_info() -> dict[str, Any]:
    """Get service information and available endpoints.
    
    Returns:
        dict: Service details and endpoints
    """
    return {
        "service": "GNN Service",
        "version": "1.0.0",
        "description": "Universal Temporal GNN for Hydraulic System Diagnostics",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "metrics": "/metrics",
            "diagnose": "/v1/diagnose",
            "predict": "/v1/predict",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
