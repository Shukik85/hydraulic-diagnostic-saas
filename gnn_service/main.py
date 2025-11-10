"""
FastAPI service for hydraulic diagnostics GNN.
Provides REST API for real-time component fault prediction.
"""

import logging
import time
from datetime import datetime
from typing import Any

import uvicorn
from config import api_config, model_config, physical_norms
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import JSONResponse  # Добавлен импорт
from inference import GNNInference, get_inference_engine
from pydantic import BaseModel, Field, validator

# Configure logging
logging.basicConfig(
    level=getattr(logging, api_config.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Hydraulic Diagnostics GNN Service",
    description="Temporal Graph Attention Network for multi-label classification of excavator components",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Global inference engine
_inference_engine: GNNInference | None = None


# Pydantic models for request/response validation
class ComponentFeatures(BaseModel):
    """Features for a single component."""

    raw_features: list[float] = Field(
        ..., min_items=5, max_items=5, description="5 raw sensor values"
    )
    normalized_features: list[float] = Field(
        ..., min_items=5, max_items=5, description="5 normalized values (0-1)"
    )
    deviation_features: list[float] = Field(
        ..., min_items=5, max_items=5, description="5 deviation indicators"
    )

    @validator("normalized_features", "deviation_features")
    def validate_normalized_range(cls, v):
        for value in v:
            if not (0.0 <= value <= 1.0):
                raise ValueError("Normalized values must be between 0 and 1")
        return v

    @validator("deviation_features")
    def validate_deviation_range(cls, v):
        for value in v:
            if value < 0.0 or value > 2.0:
                raise ValueError("Deviation values must be between 0 and 2")
        return v


class GraphRequest(BaseModel):
    """Request model for single graph prediction."""

    pump: ComponentFeatures
    cylinder_boom: ComponentFeatures
    cylinder_stick: ComponentFeatures
    cylinder_bucket: ComponentFeatures
    motor_swing: ComponentFeatures
    motor_left: ComponentFeatures
    motor_right: ComponentFeatures
    timestamp: str | None = None
    equipment_id: str | None = "default"

    class Config:
        schema_extra = {
            "example": {
                "pump": {
                    "raw_features": [250.0, 2000.0, 60.0, 2.5, 75.0],
                    "normalized_features": [0.8, 0.5, 0.6, 0.5, 0.7],
                    "deviation_features": [0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "cylinder_boom": {
                    "raw_features": [200.0, 60.0, 50.0, 200.0, 160.0],
                    "normalized_features": [0.7, 0.5, 0.5, 0.5, 0.6],
                    "deviation_features": [0.0, 0.0, 0.0, 0.0, 0.0],
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "equipment_id": "excavator_001",
            }
        }


class BatchGraphRequest(BaseModel):
    """Request model for batch prediction."""

    graphs: list[GraphRequest] = Field(..., max_items=api_config.max_batch_size)
    batch_id: str | None = None


class ComponentDiagnostics(BaseModel):
    """Diagnostics results for a single component."""

    fault_probability: float = Field(..., ge=0.0, le=1.0)
    status: str = Field(..., regex="^(normal|warning|critical)$")
    deviations: dict[str, dict[str, float]]
    expected_values: dict[str, float]
    attention_score: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""

    system_health: float = Field(..., ge=0.0, le=1.0)
    components: dict[str, ComponentDiagnostics]
    attention_analysis: dict[str, Any]
    root_cause: dict[str, Any]
    processing_time_ms: float
    timestamp: str
    equipment_id: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""

    predictions: list[PredictionResponse]
    batch_id: str
    total_processing_time_ms: float
    average_processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    service: str
    version: str
    model_loaded: bool
    timestamp: str
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    detail: str | None = None
    timestamp: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global _inference_engine
    try:
        logger.info("Starting Hydraulic Diagnostics GNN Service...")
        start_time = time.time()

        _inference_engine = get_inference_engine()

        startup_time = time.time() - start_time
        logger.info(f"Service started successfully in {startup_time:.2f} seconds")

    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Hydraulic Diagnostics GNN Service...")


# Utility functions
def get_inference_engine() -> GNNInference:
    """Get the inference engine instance."""
    if _inference_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return _inference_engine


def convert_request_to_features(request: GraphRequest) -> list[list[float]]:
    """Convert API request to node features format."""
    node_features = []

    for component in model_config.component_names:
        comp_data = getattr(request, component)
        # Combine all features: 5 raw + 5 normalized + 5 deviation = 15 features
        features = (
            comp_data.raw_features
            + comp_data.normalized_features
            + comp_data.deviation_features
        )
        node_features.append(features)

    return node_features


# API endpoints
@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirects to docs."""
    return {"message": "Hydraulic Diagnostics GNN Service", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    engine = get_inference_engine()

    return HealthResponse(
        status="healthy",
        service="hydraulic_diagnostics_gnn",
        version="1.0.0",
        model_loaded=engine is not None,
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=time.time() - getattr(app, "_startup_time", time.time()),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict(request: GraphRequest, background_tasks: BackgroundTasks = None):
    """
    Perform hydraulic system diagnostics for a single graph.

    Args:
        request: Graph features for all 7 components
        background_tasks: FastAPI background tasks (unused)

    Returns:
        Comprehensive diagnostics including fault probabilities, status, and explanations
    """
    start_time = time.time()

    try:
        engine = get_inference_engine()

        # Convert request to model input format
        node_features = convert_request_to_features(request)

        # Perform prediction
        result = engine.predict(node_features)

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Prepare response
        response = PredictionResponse(
            system_health=result["system_health"],
            components={
                comp: ComponentDiagnostics(**data)
                for comp, data in result["components"].items()
            },
            attention_analysis=result["attention_analysis"],
            root_cause=result["root_cause"],
            processing_time_ms=processing_time,
            timestamp=request.timestamp or datetime.utcnow().isoformat(),
            equipment_id=request.equipment_id,
        )

        logger.info(
            f"Prediction completed for {request.equipment_id} in {processing_time:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
    responses={
        503: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def batch_predict(request: BatchGraphRequest):
    """
    Perform batch diagnostics for multiple graphs.

    Args:
        request: List of graph requests

    Returns:
        Batch diagnostics results with aggregated timing information
    """
    start_time = time.time()

    try:
        engine = get_inference_engine()

        if len(request.graphs) > api_config.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=f"Batch size exceeds maximum of {api_config.max_batch_size}",
            )

        # Convert requests to model input format
        batch_node_features = [
            convert_request_to_features(graph) for graph in request.graphs
        ]

        # Perform batch prediction
        results = engine.batch_predict(batch_node_features)

        total_processing_time = (time.time() - start_time) * 1000
        avg_processing_time = total_processing_time / len(request.graphs)

        # Prepare responses
        prediction_responses = []
        for i, result in enumerate(results):
            if "error" in result:
                # Skip errored results
                continue

            prediction_responses.append(
                PredictionResponse(
                    system_health=result["system_health"],
                    components={
                        comp: ComponentDiagnostics(**data)
                        for comp, data in result["components"].items()
                    },
                    attention_analysis=result["attention_analysis"],
                    root_cause=result["root_cause"],
                    processing_time_ms=avg_processing_time,  # Approximate
                    timestamp=request.graphs[i].timestamp
                    or datetime.utcnow().isoformat(),
                    equipment_id=request.graphs[i].equipment_id,
                )
            )

        response = BatchPredictionResponse(
            predictions=prediction_responses,
            batch_id=request.batch_id or f"batch_{int(time.time())}",
            total_processing_time_ms=total_processing_time,
            average_processing_time_ms=avg_processing_time,
        )

        logger.info(
            f"Batch prediction completed: {len(prediction_responses)} graphs "
            f"in {total_processing_time:.2f}ms"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/config")
async def get_model_config():
    """Get current model configuration."""
    engine = get_inference_engine()

    return {
        "model_config": {
            "num_node_features": model_config.num_node_features,
            "hidden_dim": model_config.hidden_dim,
            "num_classes": model_config.num_classes,
            "num_gat_layers": model_config.num_gat_layers,
            "num_heads": model_config.num_heads,
            "component_names": model_config.component_names,
        },
        "thresholds": {
            "warning": engine.warning_threshold,
            "critical": engine.critical_threshold,
        },
        "physical_norms": {
            component: getattr(physical_norms, component.upper())
            for component in model_config.component_names
        },
    }


@app.put("/model/thresholds")
async def update_thresholds(warning: float = None, critical: float = None):
    """
    Update classification thresholds.

    Args:
        warning: New warning threshold (0-1)
        critical: New critical threshold (0-1)
    """
    engine = get_inference_engine()

    if warning is not None and not (0 <= warning <= 1):
        raise HTTPException(
            status_code=422, detail="Warning threshold must be between 0 and 1"
        )

    if critical is not None and not (0 <= critical <= 1):
        raise HTTPException(
            status_code=422, detail="Critical threshold must be between 0 and 1"
        )

    if warning is not None and critical is not None and warning >= critical:
        raise HTTPException(
            status_code=422, detail="Warning threshold must be less than critical"
        )

    engine.update_thresholds(warning, critical)

    return {
        "message": "Thresholds updated successfully",
        "new_thresholds": {
            "warning": engine.warning_threshold,
            "critical": engine.critical_threshold,
        },
    }


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail, timestamp=datetime.utcnow().isoformat()
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat(),
        ).dict(),
    )


# Middleware for logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests and responses."""
    start_time = time.time()

    response = await call_next(request)

    processing_time = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {processing_time:.2f}ms"
    )

    return response


if __name__ == "__main__":
    # Store startup time for uptime calculation
    app._startup_time = time.time()

    # Run the application
    uvicorn.run(
        app,
        host=api_config.host,
        port=api_config.port,
        log_level=api_config.log_level,
        reload=api_config.debug,
    )
