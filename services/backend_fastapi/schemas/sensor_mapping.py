"""
Pydantic schemas for sensor mapping
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid


class SensorMappingCreate(BaseModel):
    equipment_id: uuid.UUID
    component_index: int = Field(ge=0)
    sensor_id: str = Field(min_length=1, max_length=255)
    sensor_type: str  # "pressure", "temperature", "vibration", "flow"
    expected_range_min: Optional[float] = None
    expected_range_max: Optional[float] = None
    unit: str
    sampling_rate: Optional[int] = None
    is_critical: bool = False
    description: Optional[str] = None


class SensorMappingUpdate(BaseModel):
    expected_range_min: Optional[float] = None
    expected_range_max: Optional[float] = None
    unit: Optional[str] = None
    is_critical: Optional[bool] = None
    description: Optional[str] = None


class SensorMappingResponse(BaseModel):
    id: uuid.UUID
    equipment_id: uuid.UUID
    component_index: int
    component_name: str
    sensor_id: str
    sensor_type: str
    expected_range_min: Optional[float]
    expected_range_max: Optional[float]
    unit: str
    auto_detected: bool
    confidence_score: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


class SensorSuggestion(BaseModel):
    component_index: int
    component_name: str
    sensor_id: str
    sensor_type: str
    expected_range_min: float
    expected_range_max: float
    unit: str
    confidence: float
    auto_detected: bool


class AutoDetectResponse(BaseModel):
    equipment_id: uuid.UUID
    total_sensors: int
    matched_sensors: int
    suggestions: List[SensorSuggestion]
