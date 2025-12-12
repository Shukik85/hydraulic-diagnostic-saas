"""GNN Service FastAPI Application.

Production-ready API for hydraulic system diagnostics using GNN.

Endpoints:
    v2 (Phase 3.1):
        - POST /api/v2/inference/minimal
        - GET /api/v2/topologies
        - GET /api/v2/topologies/{topology_id}
        - POST /api/v2/topologies/validate
    
    v1 (Legacy):
        - POST /api/v1/predict
        - POST /api/v1/batch/predict
    
    Health:
        - GET /health
        - GET /healthz
        - GET /ready

Author: GNN Service Team
Python: 3.14+
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.schemas import BatchPredictionRequest, GraphTopology, PredictionRequest, PredictionResponse
from src.schemas.requests import MinimalInferenceRequest
from src.services.topology_service import TopologyService, get_topology_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global instances (initialized in lifespan)
engine: InferenceEngine | None = None
topology_service: TopologyService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown.
    
    Initializes:
    - InferenceEngine
    - TopologyService
    - Model loading
    """
    global engine, topology_service

    logger.info("Starting GNN Service...")

    # Load configuration
    model_path = os.getenv("MODEL_PATH", "models/v2.0.0.ckpt")
    device = os.getenv("DEVICE", "auto")
    batch_size = int(os.getenv("BATCH_SIZE", "32"))

    # Initialize inference engine
    try:
        config = InferenceConfig(
            model_path=model_path,
            device=device,
            batch_size=batch_size,
            use_dynamic_features=True  # Phase 3.1
        )

        engine = InferenceEngine(config=config)
        logger.info(f"InferenceEngine initialized (device={device})")
    except Exception as e:
        logger.error(f"Failed to initialize InferenceEngine: {e}")
        raise

    # Initialize topology service
    try:
        topology_service = get_topology_service()
        logger.info("TopologyService initialized")
    except Exception as e:
        logger.error(f"Failed to initialize TopologyService: {e}")
        raise

    logger.info("GNN Service started successfully")

    yield

    # Shutdown
    logger.info("Shutting down GNN Service...")
    engine = None
    topology_service = None


# Create FastAPI app
app = FastAPI(
    title="GNN Hydraulic Diagnostics Service",
    description="Production GNN service for hydraulic system diagnostics",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Health Check Endpoints
# ============================================================================

@app.get("/health", tags=["Health"])
@app.get("/healthz", tags=["Health"])
async def health_check():
    """Basic health check.
    
    Returns:
        Status: healthy if service is running
    """
    return {
        "status": "healthy",
        "service": "gnn-service",
        "version": "2.0.0"
    }


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check (model loaded, services ready).
    
    Returns:
        Status: ready if all components initialized
    """
    if engine is None or topology_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )

    return {
        "status": "ready",
        "model_loaded": engine is not None,
        "topology_service_ready": topology_service is not None,
        "stats": engine.get_stats() if engine else {}
    }


# ============================================================================
# v2 API Endpoints (Phase 3.1)
# ============================================================================

@app.post(
    "/api/v2/inference/minimal",
    response_model=PredictionResponse,
    tags=["Inference v2"]
)
async def predict_minimal(request: MinimalInferenceRequest) -> PredictionResponse:
    """Minimal inference endpoint (Phase 3.1).
    
    Simplest API - only requires:
    - equipment_id
    - timestamp
    - sensor_readings (per component)
    - topology_id (template name)
    
    Dynamic edge features are auto-computed from sensor readings.
    
    Args:
        request: MinimalInferenceRequest
    
    Returns:
        PredictionResponse with health, degradation, anomaly predictions
    
    Example:
        ```json
        {
          "equipment_id": "pump_system_01",
          "timestamp": "2025-12-03T23:00:00Z",
          "sensor_readings": {
            "pump_1": {
              "pressure_bar": 150.0,
              "temperature_c": 65.0,
              "vibration_g": 0.8
            },
            "valve_1": {
              "pressure_bar": 148.0,
              "temperature_c": 64.0
            }
          },
          "topology_id": "standard_pump_system"
        }
        ```
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready"
        )

    try:
        response = await engine.predict_minimal(request)
        return response

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.get(
    "/api/v2/topologies",
    tags=["Topology v2"]
)
async def list_topologies():
    """List all available topology templates.
    
    Returns:
        List of template metadata (id, name, description, components, edges)
    
    Example response:
        ```json
        {
          "templates": [
            {
              "template_id": "standard_pump_system",
              "name": "Standard Pump System",
              "description": "...",
              "num_components": 4,
              "num_edges": 3
            }
          ]
        }
        ```
    """
    if topology_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Topology service not ready"
        )

    try:
        templates = topology_service.list_templates()
        return {"templates": templates}

    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list templates"
        )


@app.get(
    "/api/v2/topologies/{topology_id}",
    tags=["Topology v2"]
)
async def get_topology(topology_id: str):
    """Get topology template by ID.
    
    Args:
        topology_id: Template identifier
    
    Returns:
        Template details (components, edges, metadata)
    
    Raises:
        404: Template not found
    """
    if topology_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Topology service not ready"
        )

    try:
        template = topology_service.get_template(topology_id)

        if template is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topology not found: {topology_id}"
            )

        return template.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get template {topology_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get template"
        )


@app.post(
    "/api/v2/topologies/validate",
    tags=["Topology v2"]
)
async def validate_topology(topology: GraphTopology):
    """Validate custom topology.
    
    Checks:
    - All edges reference existing components
    - No duplicate component IDs
    - Edge properties are valid
    
    Args:
        topology: Custom GraphTopology
    
    Returns:
        Validation result (is_valid, errors)
    
    Example:
        ```json
        {
          "equipment_id": "custom_system",
          "components": {...},
          "edges": [...]
        }
        ```
    """
    if topology_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Topology service not ready"
        )

    try:
        is_valid, errors = topology_service.validate_topology(topology)

        return {
            "is_valid": is_valid,
            "errors": errors,
            "num_components": len(topology.components),
            "num_edges": len(topology.edges)
        }

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Validation failed"
        )


# ============================================================================
# v1 API Endpoints (Legacy - Backward Compatible)
# ============================================================================

@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    tags=["Inference v1 (Legacy)"]
)
async def predict_legacy(
    request: PredictionRequest,
    topology: GraphTopology
) -> PredictionResponse:
    """Legacy single prediction endpoint.
    
    Backward compatible with v1 API.
    
    Args:
        request: PredictionRequest
        topology: GraphTopology
    
    Returns:
        PredictionResponse
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready"
        )

    try:
        response = await engine.predict(request, topology)
        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed"
        )


@app.post(
    "/api/v1/batch/predict",
    response_model=list[PredictionResponse],
    tags=["Inference v1 (Legacy)"]
)
async def predict_batch_legacy(
    batch_request: BatchPredictionRequest,
    topology: GraphTopology
) -> list[PredictionResponse]:
    """Legacy batch prediction endpoint.
    
    Backward compatible with v1 API.
    
    Args:
        batch_request: BatchPredictionRequest
        topology: GraphTopology
    
    Returns:
        List of PredictionResponse
    """
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready"
        )

    try:
        responses = await engine.predict_batch(
            requests=batch_request.requests,
            topology=topology
        )
        return responses

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed"
        )


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle Pydantic validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Validation error",
            "errors": exc.errors()
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set True for development
        log_level="info"
    )
