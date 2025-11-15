"""Standalone test_inference endpoint для запуска inference без deply в prod."""
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional

router = APIRouter(prefix="/admin", tags=["Admin"])

class TestInferenceRequest(BaseModel):
    equipment_id: str = Field(...)
    time_window: Dict = Field(...)
    model_path: str = Field(..., description="Путь к тестовой .onnx модели (НЕ прод)")
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
    import time
    from inference_dynamic import run_dynamic_inference
    from onnxruntime import InferenceSession
    import os
    import datetime
    
    # Проверяем, что model_path существует и это не путь до prod
    model_path = Path(request.model_path)
    prod_path = Path("/app/models/current/model.onnx")
    if not model_path.exists() or model_path == prod_path:
        raise HTTPException(status_code=400, detail="Wrong model_path (does not exist or points to prod)")
    
    # Проверка модели (ONNX)
    try:
        sess = InferenceSession(str(model_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot load ONNX model: {e}")
    
    start_time = time.time()
    request_id = f"test_{int(start_time * 1000)}"
    
    # Test-inference (использует выбранную модель)
    result = await run_dynamic_inference(
        equipment_id=request.equipment_id,
        time_window=request.time_window,
        sensor_data=request.sensor_data,
        model_override_path=str(model_path)
    )
    inference_time = (time.time() - start_time) * 1000
    return TestInferenceResponse(
        request_id=request_id,
        overall_health_score=result["overall_health_score"],
        component_health=result["component_health"],
        anomalies=result["anomalies"],
        recommendations=result["recommendations"],
        inference_time_ms=inference_time,
        timestamp=datetime.datetime.utcnow().isoformat(),
        model_version=f"test_{os.path.basename(model_path)}",
        from_prod=False
    )
