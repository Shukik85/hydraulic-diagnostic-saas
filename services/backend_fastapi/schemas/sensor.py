"""
Sensor data ingestion schemas
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from datetime import datetime
from uuid import UUID


class SensorReading(BaseModel):
    """Single sensor reading"""
    component_id: str
    sensor_name: str
    value: float
    unit: Optional[str] = None
    timestamp: datetime


class SensorDataIngest(BaseModel):
    """Batch sensor data ingestion"""
    system_id: str
    readings: List[SensorReading] = Field(..., min_length=1)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "system_id": "press_01",
            "readings": [
                {
                    "component_id": "pump",
                    "sensor_name": "pressure",
                    "value": 150.5,
                    "unit": "bar",
                    "timestamp": "2025-11-11T12:00:00Z"
                }
            ]
        }
    })


class SensorDataResponse(BaseModel):
    """Response after data ingestion"""
    ingested_count: int
    quarantined_count: int
    errors: List[str] = []
    ingestion_id: UUID
