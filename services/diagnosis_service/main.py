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
