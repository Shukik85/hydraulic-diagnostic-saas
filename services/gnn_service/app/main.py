"""FastAPI application for Hydraulic Diagnostics Service.

Modern REST API providing:
- Real-time equipment diagnostics
- Multi-component health predictions
- System health monitoring

Features:
    - Async request handling (Python 3.14+)
    - Pydantic request/response validation
    - CORS middleware for frontend integration
    - Structured logging
    - OpenAPI documentation
    - Health checks

Endpoints:
    POST /api/v1/diagnostics/predict - Predict equipment health
    GET  /api/v1/health - Service health check
    GET  /docs - Swagger UI documentation
    GET  /redoc - ReDoc documentation

Usage:
    from app.main import app
    import uvicorn
    
    # Development
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    
    # Production
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)

Python 3.14 Features:
    - Deferred annotations (__future__)
    - Union types
    - Async/await
    - Pattern matching (future)
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    SensorData,
    ComponentPrediction,
    DiagnosticResponse,
    HealthResponse
)
from app.inference_mock import MockInferenceEngine

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

# ============================================================================
# APPLICATION SETUP
# ============================================================================

app = FastAPI(
    title="Hydraulic Diagnostics GNN Service",
    description="Real-time hydraulic system diagnostics using Graph Neural Networks",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# ============================================================================
# GLOBAL STATE
# ============================================================================

inference_engine: MockInferenceEngine | None = None
service_start_time: float = time.time()


# ============================================================================
# STARTUP/SHUTDOWN EVENTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    global inference_engine
    
    logger.info("🚀 Starting Hydraulic Diagnostics Service...")
    
    try:
        # Initialize mock inference engine
        inference_engine = MockInferenceEngine()
        logger.info("✅ Inference engine initialized")
        
        logger.info("🚀 Service started successfully!")
        logger.info(f"   API Documentation: http://localhost:8000/docs")
        logger.info(f"   Health Check: http://localhost:8000/api/v1/health")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("🛑 Shutting down service...")
    # Add cleanup code here if needed


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An error occurred",
            "type": exc.__class__.__name__
        }
    )


# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - service information."""
    return {
        "service": "Hydraulic Diagnostics GNN",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/api/v1/health"
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Service health check.
    
    Returns:
        HealthResponse with service status and details.
    
    Examples:
        GET /api/v1/health
        
        Response:
        {
            "status": "healthy",
            "model_loaded": true,
            "cuda_available": false,
            "version": "mock-v0.1.0"
        }
    """
    global inference_engine
    
    import torch
    
    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        cuda_available=torch.cuda.is_available(),
        version=inference_engine.model_version if inference_engine else "unknown"
    )


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@app.post("/api/v1/diagnostics/predict", response_model=DiagnosticResponse, tags=["Diagnostics"])
async def predict_diagnostics(request_data: SensorData):
    """Predict hydraulic system health from sensor data.
    
    Performs real-time diagnostics using GNN model:
    1. Validates sensor data
    2. Extracts features
    3. Builds dynamic graph
    4. Runs GNN inference
    5. Returns component predictions
    
    Args:
        request_data: Sensor readings and equipment info
    
    Returns:
        DiagnosticResponse with health predictions and recommendations
    
    Raises:
        HTTPException: If validation fails or inference errors
    
    Examples:
        POST /api/v1/diagnostics/predict
        
        Request Body:
        {
            "equipment_id": "pump_001",
            "sensor_readings": {
                "PS1": [100.5, 101.2, 100.8, ...],
                "TS1": [45.3, 45.5, 45.4, ...],
                "FS1": [8.5, 8.6, 8.5, ...]
            },
            "lookback_minutes": 10
        }
        
        Response:
        {
            "equipment_id": "pump_001",
            "timestamp": "2025-12-12T12:00:00Z",
            "overall_health": 0.85,
            "components": [
                {
                    "component_name": "pump",
                    "health_score": 0.85,
                    "severity_grade": "optimal",
                    "confidence": 0.92,
                    "contributing_sensors": ["PS1", "FS1"]
                },
                ...
            ],
            "recommendations": ["✅ All components operating optimally"],
            "model_version": "mock-v0.1.0",
            "inference_time_ms": 52.34
        }
    """
    global inference_engine
    
    start_time = time.time()
    
    # Validate service is ready
    if inference_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready - inference engine not initialized"
        )
    
    try:
        # Validate sensor data
        if not request_data.sensor_readings:
            raise HTTPException(
                status_code=400,
                detail="sensor_readings cannot be empty"
            )
        
        # Check all sensors have same number of samples
        sample_counts = [len(v) for v in request_data.sensor_readings.values()]
        if len(set(sample_counts)) > 1:
            raise HTTPException(
                status_code=400,
                detail=f"All sensors must have same number of samples. Got: {sample_counts}"
            )
        
        logger.info(
            f"Predicting for {request_data.equipment_id}: "
            f"{len(request_data.sensor_readings)} sensors, "
            f"{sample_counts[0] if sample_counts else 0} samples"
        )
        
        # Run inference
        predictions = await inference_engine.predict(
            equipment_id=request_data.equipment_id,
            sensor_readings=request_data.sensor_readings,
            topology_id=request_data.topology_id
        )
        
        # Calculate inference time
        inference_time_ms = (time.time() - start_time) * 1000
        
        # Build response
        return DiagnosticResponse(
            equipment_id=request_data.equipment_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_health=predictions["overall_health"],
            components=[
                ComponentPrediction(**c) for c in predictions["components"]
            ],
            recommendations=predictions["recommendations"],
            model_version=inference_engine.model_version,
            inference_time_ms=round(inference_time_ms, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting Hydraulic Diagnostics API...")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
