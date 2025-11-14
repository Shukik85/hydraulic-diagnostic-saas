<<<<<<< HEAD
# services/diagnosis_service/main.py
"""
Diagnosis Service - Orchestrates full diagnosis pipeline.
"""
import os
import logging
from typing import Dict, List, Optional
from datetime import datetime
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import httpx

from openapi_config import custom_openapi
from monitoring_endpoints import router as monitoring_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Diagnosis Service API",
    version="1.0.0",
    description="Diagnosis orchestration service",
    openapi_version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(monitoring_router)
app.openapi = lambda: custom_openapi(app)


# === Models ===

class TimeWindow(BaseModel):
    """Time range for diagnosis."""
    start_time: str = Field(..., description="ISO 8601 timestamp")
    end_time: str = Field(..., description="ISO 8601 timestamp")


class DiagnosisRequest(BaseModel):
    """Diagnosis request."""
    equipment_id: str = Field(..., description="Equipment to diagnose")
    time_window: TimeWindow
    include_rag: bool = Field(True, description="Include RAG interpretation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "equipment_id": "exc_001",
                "time_window": {
                    "start_time": "2025-11-01T00:00:00Z",
                    "end_time": "2025-11-13T00:00:00Z"
                },
                "include_rag": True
            }
        }


class DiagnosisResponse(BaseModel):
    """Complete diagnosis result."""
    diagnosis_id: str
    equipment_id: str
    status: str  # "pending" | "processing" | "completed" | "failed"
    gnn_result: Optional[Dict] = None
    rag_interpretation: Optional[Dict] = None
    created_at: str
    completed_at: Optional[str] = None
    progress: float = Field(0.0, ge=0.0, le=100.0)


# === Endpoints ===

@app.post("/diagnosis", response_model=DiagnosisResponse, tags=["Diagnosis"])
async def run_diagnosis(
    request: DiagnosisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start full diagnosis pipeline.
    
    **Process**:
    1. Query sensor data from TimescaleDB
    2. Call GNN Service for inference
    3. Call RAG Service for interpretation
    4. Store results
    
    **Returns**: Diagnosis ID (poll /diagnosis/{id} for results)
    
    **Latency**: 5-10 seconds total (async)
    """
    try:
        diagnosis_id = str(uuid.uuid4())
        
        logger.info(f"Starting diagnosis {diagnosis_id} for {request.equipment_id}")
        
        # Start async processing
        background_tasks.add_task(
            process_diagnosis,
            diagnosis_id,
            request.equipment_id,
            request.time_window,
            request.include_rag
        )
        
        return DiagnosisResponse(
            diagnosis_id=diagnosis_id,
            equipment_id=request.equipment_id,
            status="processing",
            created_at=datetime.utcnow().isoformat(),
            progress=0.0
        )
        
    except Exception as e:
        logger.error(f"Failed to start diagnosis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/diagnosis/{diagnosis_id}", response_model=DiagnosisResponse, tags=["Diagnosis"])
async def get_diagnosis(diagnosis_id: str):
    """
    Get diagnosis results.
    
    **Poll this endpoint** for async results.
    """
    try:
        # TODO: Fetch from database
        logger.info(f"Fetching diagnosis: {diagnosis_id}")
        
        return DiagnosisResponse(
            diagnosis_id=diagnosis_id,
            equipment_id="exc_001",
            status="completed",
            created_at=datetime.utcnow().isoformat(),
            progress=100.0
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="Diagnosis not found")


@app.get("/diagnosis/{diagnosis_id}/progress", tags=["Diagnosis"])
async def get_diagnosis_progress(diagnosis_id: str):
    """
    Get real-time diagnosis progress.
    
    **Use case**: WebSocket alternative for progress tracking.
    """
    try:
        # TODO: Fetch progress from cache/database
        return {
            "diagnosis_id": diagnosis_id,
            "progress": 65.0,
            "current_step": "rag_interpretation",
            "estimated_completion_seconds": 2
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="Diagnosis not found")


@app.post("/diagnosis/batch", tags=["Diagnosis"])
async def batch_diagnosis(requests: List[DiagnosisRequest]):
    """
    Batch diagnosis for multiple equipment.
    
    **Max**: 50 per batch
    """
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Max 50 requests per batch")
    
    try:
        diagnosis_ids = []
        
        for req in requests:
            result = await run_diagnosis(req, BackgroundTasks())
            diagnosis_ids.append(result.diagnosis_id)
        
        return {
            "batch_size": len(diagnosis_ids),
            "diagnosis_ids": diagnosis_ids
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "service": "Diagnosis Service",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


# === Background Processing ===

async def process_diagnosis(
    diagnosis_id: str,
    equipment_id: str,
    time_window: TimeWindow,
    include_rag: bool
):
    """
    Background task для processing diagnosis.
    """
    try:
        logger.info(f"Processing diagnosis {diagnosis_id}")
        
        # 1. Call GNN Service
        async with httpx.AsyncClient() as client:
            gnn_response = await client.post(
                "http://gnn-service:8002/inference",
                json={
                    "equipment_id": equipment_id,
                    "time_window": time_window.dict()
                },
                timeout=30.0
            )
            gnn_result = gnn_response.json()
        
        # 2. Call RAG Service (if enabled)
        rag_result = None
        if include_rag:
            async with httpx.AsyncClient() as client:
                rag_response = await client.post(
                    "http://rag-service:8004/interpret/diagnosis",
                    json={
                        "gnn_result": gnn_result,
                        "equipment_context": {"equipment_id": equipment_id}
                    },
                    timeout=30.0
                )
                rag_result = rag_response.json()
        
        # 3. Store results
        # TODO: Save to database
        
        logger.info(f"Diagnosis {diagnosis_id} completed")
        
    except Exception as e:
        logger.error(f"Diagnosis processing failed: {e}")
        # TODO: Update status to failed


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
=======
"""
Diagnosis Service - Updated to use shared package
"""

from fastapi import FastAPI, HTTPException
from typing import List
import httpx

Import from shared package
from shared.clients import GNNClient, GNNPredictionRequest
from shared.schemas import SensorData
from shared.validation import validate_sensor_batch

app = FastAPI(
title="Diagnosis Service",
version="1.0.0",
description="Orchestrates diagnosis pipeline using shared utilities"
)

Initialize shared GNN client
gnn_client = GNNClient(base_url="http://gnn-service:8001")

@app.get("/health")
async def health():
"""Health check."""
return {
"service": "diagnosis-service",
"status": "healthy",
"shared_package": "enabled"
}

@app.post("/diagnose")
async def run_diagnosis(sensor_data: List[SensorData]):
"""
Run complete diagnosis pipeline.

text
Uses:
- shared.schemas.SensorData for input validation
- shared.validation.validate_sensor_batch for validation
- shared.clients.GNNClient for ML inference
"""

# Step 1: Validate sensor data (using shared validation)
validation_result = validate_sensor_batch(sensor_data)

if validation_result['invalid_count'] > 0:
    raise HTTPException(
        status_code=400,
        detail={
            "error": "Invalid sensor data detected",
            "invalid_sensors": validation_result['invalid'],
            "validation": validation_result
        }
    )

# Step 2: Call Equipment Service for context
async with httpx.AsyncClient() as client:
    equipment_response = await client.get(
        "http://equipment-service:8002/systems"
    )
    equipment_context = equipment_response.json()

# Step 3: Call GNN Service (using shared client)
gnn_request = GNNPredictionRequest(
    sensor_data=[s.model_dump() for s in sensor_data],
    timestamp=sensor_data.timestamp.isoformat()
)

try:
    gnn_result = await gnn_client.predict(gnn_request)
except Exception as e:
    raise HTTPException(
        status_code=503,
        detail=f"GNN Service unavailable: {str(e)}"
    )

# Step 4: Call RAG Service for interpretation
async with httpx.AsyncClient() as client:
    rag_response = await client.post(
        "http://rag-service:8004/interpret",
        json={
            "predictions": gnn_result.predictions,
            "confidence": gnn_result.confidence,
            "context": equipment_context
        }
    )
    rag_interpretation = rag_response.json()

# Step 5: Aggregate and return
return {
    "status": "success",
    "validation": validation_result,
    "gnn_analysis": {
        "predictions": gnn_result.predictions,
        "confidence": gnn_result.confidence,
        "attention_weights": gnn_result.attention_weights,
        "inference_time_ms": gnn_result.inference_time_ms
    },
    "interpretation": rag_interpretation,
    "equipment_context": equipment_context
}
@app.on_event("shutdown")
async def shutdown():
"""Close GNN client on shutdown."""
await gnn_client.close()

if name == "main":
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8003)
>>>>>>> da16424e28634340833b5d4c6ea96234fc52ac16
