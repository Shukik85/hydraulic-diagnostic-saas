# services/gnn_service/main.py
"""
GNN Service - Complete implementation.
Graph Neural Network inference, model management, training.
"""
<<<<<<< HEAD
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from openapi_config import custom_openapi
from admin_endpoints import router as admin_router

=======

import logging
import os
from datetime import datetime

import uvicorn
from admin_endpoints import router as admin_router
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openapi_config import custom_openapi
from pydantic import BaseModel, Field

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GNN Service API",
    version="1.0.0",
    description="Graph Neural Network inference and model management",
    openapi_version="3.1.0",
    docs_url="/docs",
<<<<<<< HEAD
    redoc_url="/redoc"
=======
    redoc_url="/redoc",
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include admin router
app.include_router(admin_router)

# Apply custom OpenAPI
app.openapi = lambda: custom_openapi(app)


# === Request/Response Models ===

<<<<<<< HEAD
class InferenceRequest(BaseModel):
    """Single inference request."""
    equipment_id: str = Field(..., description="Equipment ID")
    time_window: Dict = Field(..., description="Time range for data")
    sensor_data: Optional[Dict] = Field(None, description="Pre-fetched sensor data")
    
=======

class InferenceRequest(BaseModel):
    """Single inference request."""

    equipment_id: str = Field(..., description="Equipment ID")
    time_window: dict = Field(..., description="Time range for data")
    sensor_data: dict | None = Field(None, description="Pre-fetched sensor data")

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "exc_001",
                "time_window": {
                    "start_time": "2025-11-01T00:00:00Z",
<<<<<<< HEAD
                    "end_time": "2025-11-13T00:00:00Z"
                }
=======
                    "end_time": "2025-11-13T00:00:00Z",
                },
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
            }
        }


class ComponentHealth(BaseModel):
    """Component health status."""
<<<<<<< HEAD
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    component_id: str
    component_type: str
    health_score: float = Field(..., ge=0.0, le=1.0)
    degradation_rate: float
    confidence: float = Field(..., ge=0.0, le=1.0)


class Anomaly(BaseModel):
    """Detected anomaly."""
<<<<<<< HEAD
    anomaly_type: str
    severity: str  # "low" | "medium" | "high" | "critical"
    confidence: float
    affected_components: List[str]
=======

    anomaly_type: str
    severity: str  # "low" | "medium" | "high" | "critical"
    confidence: float
    affected_components: list[str]
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    description: str


class InferenceResponse(BaseModel):
    """GNN inference result."""
<<<<<<< HEAD
    request_id: str
    overall_health_score: float = Field(..., ge=0.0, le=1.0)
    component_health: List[ComponentHealth]
    anomalies: List[Anomaly]
    recommendations: List[str]
=======

    request_id: str
    overall_health_score: float = Field(..., ge=0.0, le=1.0)
    component_health: list[ComponentHealth]
    anomalies: list[Anomaly]
    recommendations: list[str]
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    inference_time_ms: float
    timestamp: str
    model_version: str


class BatchInferenceRequest(BaseModel):
    """Batch inference request."""
<<<<<<< HEAD
    requests: List[InferenceRequest] = Field(..., max_length=100)
=======

    requests: list[InferenceRequest] = Field(..., max_length=100)
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16


class ModelInfoResponse(BaseModel):
    """Model information."""
<<<<<<< HEAD
    model_version: str
    model_type: str
    input_shape: List[int]
=======

    model_version: str
    model_type: str
    input_shape: list[int]
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    output_classes: int
    framework: str
    deployed_at: str


# === Monitoring Endpoints ===

<<<<<<< HEAD
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
@app.get("/health", tags=["Monitoring"])
async def health_check():
    """
    Service health check.
<<<<<<< HEAD
    
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    Checks:
    - Service is running
    - Model is loaded
    - GPU available (if configured)
    """
    try:
        # Check model loaded
        # model = get_model()
<<<<<<< HEAD
        
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        return {
            "service": "gnn-service",
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "checks": {
                "model": "loaded",
<<<<<<< HEAD
                "gpu": "available"  # TODO: actual check
            }
        }
    except Exception as e:
        return {
            "service": "gnn-service",
            "status": "unhealthy",
            "error": str(e)
        }
=======
                "gpu": "available",  # TODO: actual check
            },
        }
    except Exception as e:
        return {"service": "gnn-service", "status": "unhealthy", "error": str(e)}
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16


@app.get("/ready", tags=["Monitoring"])
async def readiness_check():
    """Readiness probe for Kubernetes."""
    return {"status": "ready"}


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    """
<<<<<<< HEAD
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
=======
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16


# === Inference Endpoints ===

<<<<<<< HEAD
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
@app.post("/inference", response_model=InferenceResponse, tags=["Inference"])
async def run_inference(request: InferenceRequest):
    """
    Run GNN inference on equipment data.
<<<<<<< HEAD
    
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Process**:
    1. Fetch sensor data from TimescaleDB
    2. Build graph structure
    3. Run GNN model
    4. Post-process results
    5. Generate recommendations
<<<<<<< HEAD
    
    **Returns**: Health scores, anomalies, recommendations
    
=======

    **Returns**: Health scores, anomalies, recommendations

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Latency**: ~500ms typical
    """
    try:
        import time
<<<<<<< HEAD
        from inference_dynamic import run_dynamic_inference
        
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"
        
        logger.info(f"Running inference for {request.equipment_id}")
        
=======

        from inference_dynamic import run_dynamic_inference

        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}"

        logger.info(f"Running inference for {request.equipment_id}")

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        # Run inference
        result = await run_dynamic_inference(
            equipment_id=request.equipment_id,
            time_window=request.time_window,
<<<<<<< HEAD
            sensor_data=request.sensor_data
        )
        
        inference_time = (time.time() - start_time) * 1000
        
=======
            sensor_data=request.sensor_data,
        )

        inference_time = (time.time() - start_time) * 1000

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
        return InferenceResponse(
            request_id=request_id,
            overall_health_score=result["overall_health_score"],
            component_health=result["component_health"],
            anomalies=result["anomalies"],
            recommendations=result["recommendations"],
            inference_time_ms=inference_time,
            timestamp=datetime.utcnow().isoformat(),
<<<<<<< HEAD
            model_version="1.0.0"
        )
        
=======
            model_version="1.0.0",
        )

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/batch-inference", tags=["Inference"])
async def batch_inference(request: BatchInferenceRequest):
    """
    Batch inference for multiple equipment.
<<<<<<< HEAD
    
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Max**: 100 requests per batch
    **Latency**: ~50ms per request
    """
    try:
        results = []
<<<<<<< HEAD
        
        for req in request.requests:
            result = await run_inference(req)
            results.append(result)
        
        return {
            "batch_size": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
=======

        for req in request.requests:
            result = await run_inference(req)
            results.append(result)

        return {
            "batch_size": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
        }

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    except Exception as e:
        logger.error(f"Batch inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Inference"])
async def get_model_info():
    """
    Get current production model information.
<<<<<<< HEAD
    
=======

>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    **Public endpoint** - no auth required.
    """
    return ModelInfoResponse(
        model_version="1.0.0",
        model_type="Universal Temporal GNN (GAT + LSTM)",
        input_shape=[1, 10, 32],
        output_classes=3,
<<<<<<< HEAD
        framework="ONNX",
        deployed_at="2025-11-13T00:00:00Z"
=======
        framework="ONNX",  # НЕ ИСПОЛЬЗОВАТЬ!!!
        deployed_at="2025-11-13T00:00:00Z",
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    )


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "service": "GNN Service",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
<<<<<<< HEAD
        "health": "/health"
=======
        "health": "/health",
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
