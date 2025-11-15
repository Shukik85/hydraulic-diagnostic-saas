"""
Test inference endpoint для PyTorch TorchScript моделей (без ONNX).
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional
import torch
from datetime import datetime
import time

router = APIRouter(prefix="/admin", tags=["Admin"])

class TestInferenceRequest(BaseModel):
    equipment_id: str = Field(...)
    time_window: Dict = Field(...)
    model_path: str = Field(..., description="Путь к тестовой .pt/.pth TorchScript модели")
    sensor_data: Optional[Dict] = None

class TestInferenceResponse(BaseModel):
    request_id: str
    overall_health_score: float
    component_health: list
    anomalies: list
    recommendations: list
    inference_time_ms: float
    timestamp: str
    model_version: str
    from_prod: bool

@router.post("/model/test_inference", response_model=TestInferenceResponse)
async def test_inference(request: TestInferenceRequest):
    from inference_dynamic import run_dynamic_inference
    model_path = Path(request.model_path)
    prod_path = Path("/app/models/current/model.pt")
    if not model_path.exists() or model_path.suffix not in [".pt", ".pth"] or model_path == prod_path:
        raise HTTPException(400, "Model must exist, be .pt/.pth, and not point to prod!")
    try:
        model = torch.jit.load(str(model_path))
    except Exception as e:
        raise HTTPException(500, f"Cannot load TorchScript model: {e}")
    start_time = time.time()
    request_id = f"test_{int(start_time * 1000)}"
    # Исполнительный inference
    result = await run_dynamic_inference(
        equipment_id=request.equipment_id,
        time_window=request.time_window,
        sensor_data=request.sensor_data,
        model_override=model
    )
    inference_time = (time.time() - start_time) * 1000
    return TestInferenceResponse(
        request_id=request_id,
        overall_health_score=result["overall_health_score"],
        component_health=result["component_health"],
        anomalies=result["anomalies"],
        recommendations=result["recommendations"],
        inference_time_ms=inference_time,
        timestamp=datetime.utcnow().isoformat(),
        model_version=f"test_{model_path.stem}",
        from_prod=False
    )
