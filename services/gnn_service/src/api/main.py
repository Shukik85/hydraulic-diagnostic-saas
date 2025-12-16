"""Production-Ready GNN Service FastAPI Application.

Universal Temporal GNN (GAT + LSTM) for multi-label classification
of hydraulic system component states.

Entry Point: uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Security:
    - Request size limiting (10MB max)
    - CORS strict origin checking
    - Request ID tracing
    - Async cleanup on shutdown
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.validators import RequestValidator
from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.inference.request_context import set_request_id
from src.schemas.requests import MinimalInferenceRequest, PredictionRequest

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# MIDDLEWARE
# ============================================================================


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add X-Request-ID to all requests for tracing."""

    async def dispatch(self, request: Request, call_next):
        """Add request ID to context and response headers."""
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        set_request_id(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Limit request body size to prevent DoS."""

    def __init__(self, app, max_body_size: int = 10 * 1024 * 1024):  # 10 MB default
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):
        """Check body size before processing."""
        if request.method in ["POST", "PUT", "PATCH"]:
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > self.max_body_size:
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={"error": "Request body too large"},
                )

        return await call_next(request)


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """FastAPI lifespan context manager.

    Startup:
        - Initialize inference engine
        - Initialize request validator
        - Log configuration

    Shutdown:
        - Cleanup resources
        - Drain queues
        - Release GPU memory

    Yields:
        None: Application runs between startup and shutdown
    """
    # ========================================================================
    # STARTUP
    # ========================================================================
    try:
        # Initialize inference engine
        config = InferenceConfig()
        app.state.inference_engine = InferenceEngine(config)
        logger.info("✅ Inference engine initialized")

        # Initialize request validator with limits
        app.state.validator = RequestValidator(
            inference_engine=app.state.inference_engine,
            max_batch_size=32,  # Config value
            max_graph_size=1000,  # Config value
        )
        logger.info(
            "✅ Request validator initialized",
            extra={
                "max_batch_size": 32,
                "max_graph_size": 1000,
            },
        )

    except Exception as e:
        logger.warning(
            "⚠️  Failed to initialize inference engine",
            extra={"error": str(e)},
            exc_info=True,
        )
        app.state.inference_engine = None
        app.state.validator = None

    yield

    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    if hasattr(app.state, "inference_engine") and app.state.inference_engine:
        try:
            # FIX 5: Proper async cleanup
            logger.info("🧹 Starting engine cleanup...")
            await app.state.inference_engine.cleanup()
            logger.info("✅ Inference engine cleaned up")
        except Exception as e:
            logger.error(
                "⚠️  Cleanup error",
                extra={"error": str(e)},
                exc_info=True,
            )


# ============================================================================
# APP CREATION
# ============================================================================

app = FastAPI(
    title="GNN Service",
    description="Universal Temporal GNN for Hydraulic System Diagnostics",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Health", "description": "Health and metrics checks"},
        {"name": "Inference", "description": "Prediction and diagnosis endpoints"},
        {"name": "Info", "description": "Service information"},
    ],
)


# ============================================================================
# MIDDLEWARE REGISTRATION
# ============================================================================

# FIX 4: Security - Request ID tracking
app.add_middleware(RequestIDMiddleware)

# FIX 4: Security - Body size limiting (DoS prevention)
app.add_middleware(BodySizeLimitMiddleware, max_body_size=10 * 1024 * 1024)  # 10 MB

# FIX 4: Security - CORS hardening
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        # Add production domains here:
        # "https://yourdomain.com",
        # "https://api.yourdomain.com",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Restrict to needed methods only
    allow_headers=["X-Request-ID", "Content-Type", "Authorization"],
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
    """Basic health check - service is running.

    Returns:
        dict: Status information

    Examples:
        >>> GET /health
        {"status": "healthy", "service": "gnn-service", "version": "1.0.0"}
    """
    return {
        "status": "healthy",
        "service": "gnn-service",
        "version": app.version,
    }


@app.get(
    "/ready",
    tags=["Health"],
    summary="Service readiness check",
    response_model=dict[str, Any],
)
async def readiness_check() -> dict[str, Any]:
    """Readiness check - service ready to handle requests.

    Checks if inference engine is initialized.

    Returns:
        dict: Readiness status

    Raises:
        HTTPException(503): If inference engine not initialized

    Examples:
        >>> GET /ready
        {"ready": true, "inference_engine": "initialized"}
    """
    engine = getattr(app.state, "inference_engine", None)
    if engine is None:
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

    Returns inference engine stats if available.

    Returns:
        dict: Current metrics

    Examples:
        >>> GET /metrics
        {
            "status": "ok",
            "inference_engine": {...},
            "service_version": "1.0.0"
        }
    """
    engine = getattr(app.state, "inference_engine", None)
    if engine is None:
        return {"status": "engine_not_initialized"}

    try:
        stats = engine.get_stats()
        return {
            "status": "ok",
            "inference_engine": stats,
            "service_version": app.version,
        }
    except Exception as e:
        logger.error(
            "Metrics collection failed",
            extra={"error": str(e)},
            exc_info=True,
        )
        return {"status": "error", "error": str(e)}


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

    Validates request, runs inference, and returns diagnosis.

    Args:
        request: Inference request with system readings

    Returns:
        dict: Diagnosis results with predictions

    Raises:
        HTTPException(503): If engine not initialized
        HTTPException(404): If topology not found
        HTTPException(400): If validation fails (missing sensors, wrong dimensions)
        HTTPException(413): If graph too large
        HTTPException(500): If inference fails

    Examples:
        >>> POST /v1/diagnose
        {
            "equipment_id": "excavator_001",
            "topology_id": "double_pump_v1",
            "timestamp": "2025-12-16T20:00:00Z",
            "sensor_readings": {...}
        }

        Response:
        {
            "status": "success",
            "equipment_id": "excavator_001",
            "timestamp": "2025-12-16T20:00:00Z",
            "diagnosis": {...}
        }
    """
    # Check engine availability
    engine = getattr(app.state, "inference_engine", None)
    validator = getattr(app.state, "validator", None)

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )

    logger.info(
        "Diagnosis request received",
        extra={
            "equipment_id": request.equipment_id,
            "topology_id": getattr(request, "topology_id", "unknown"),
            "num_sensors": len(request.sensor_readings),
        },
    )

    try:
        # ✅ VALIDATE REQUEST (if validator available)
        if validator and hasattr(request, "topology_id"):
            topology = await validator.validate_diagnosis_request(request)
            logger.info(
                "Request validated",
                extra={
                    "equipment_id": request.equipment_id,
                    "num_components": topology.num_components,
                },
            )

        # Run inference
        result = await engine.predict_minimal(request)

        logger.info(
            "Diagnosis completed",
            extra={
                "equipment_id": request.equipment_id,
                "status": "success",
            },
        )

        return {
            "status": "success",
            "equipment_id": request.equipment_id,
            "timestamp": request.timestamp.isoformat(),
            "diagnosis": result,
        }

    except ValueError as ve:
        logger.warning(
            "Invalid input",
            extra={
                "equipment_id": request.equipment_id,
                "error": str(ve),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid input: {str(ve)}",
        ) from ve

    except Exception as e:
        logger.error(
            "Inference error in /v1/diagnose",
            extra={
                "equipment_id": request.equipment_id,
                "error": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal inference error",
        ) from e


@app.post(
    "/v1/predict",
    tags=["Inference"],
    summary="Get predictions",
    response_model=dict[str, Any],
    status_code=status.HTTP_200_OK,
)
async def get_predictions(request: PredictionRequest) -> dict[str, Any]:
    """Get detailed predictions for system components.

    Supports batch inference with validation.

    Args:
        request: Prediction request with optional batch

    Returns:
        dict: Detailed predictions per component

    Raises:
        HTTPException(503): If engine not initialized
        HTTPException(413): If batch too large
        HTTPException(400): If validation fails
        HTTPException(500): If prediction fails

    Examples:
        >>> POST /v1/predict
        {
            "topology": {...},
            "batch": [...]
        }

        Response:
        {
            "status": "success",
            "predictions": [...]
        }
    """
    # Check engine availability
    engine = getattr(app.state, "inference_engine", None)
    validator = getattr(app.state, "validator", None)

    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not available",
        )

    logger.info(
        "Prediction request received",
        extra={
            "batch_size": len(request.batch) if hasattr(request, "batch") else 1,
        },
    )

    try:
        # ✅ VALIDATE REQUEST (if validator available)
        if validator:
            await validator.validate_prediction_request(request)
            logger.info("Batch request validated")

        # Run inference
        predictions = await engine.predict(request, request.topology)

        logger.info(
            "Predictions completed",
            extra={"status": "success"},
        )

        return {
            "status": "success",
            "predictions": predictions,
        }

    except Exception as e:
        logger.error(
            "Prediction error",
            extra={"error": str(e)},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate predictions",
        ) from e


# ============================================================================
# ERROR HANDLERS
# ============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(
    _request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle HTTP exceptions with structured response.

    Args:
        _request: FastAPI request (unused)
        exc: HTTP exception

    Returns:
        JSONResponse: Error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    _request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions.

    Logs full traceback and returns safe error message.

    Args:
        _request: FastAPI request (unused)
        exc: Unexpected exception

    Returns:
        JSONResponse: Generic error response
    """
    logger.error(
        "Unhandled exception",
        extra={"error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
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

    Examples:
        >>> GET /
        {
            "service": "GNN Service",
            "version": "1.0.0",
            "endpoints": {...}
        }
    """
    return {
        "service": "GNN Service",
        "version": app.version,
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
