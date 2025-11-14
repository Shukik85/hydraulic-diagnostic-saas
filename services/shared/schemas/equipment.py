"""
Canonical Equipment Schemas
Used across all services for equipment data.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class ComponentType(str, Enum):
    """Equipment component types."""
    PUMP = "pump"
    VALVE = "valve"
    CYLINDER = "cylinder"
    MOTOR = "motor"
    FILTER = "filter"
    ACCUMULATOR = "accumulator"


class ComponentStatus(str, Enum):
    """Component health status."""
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SensorType(str, Enum):
    """Sensor types."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW_RATE = "flow_rate"
    VIBRATION = "vibration"
    POSITION = "position"


class SensorData(BaseModel):
    """Single sensor reading."""
    
    sensor_id: str = Field(..., min_length=1, max_length=50)
    sensor_type: SensorType
    timestamp: datetime
    value: float = Field(..., ge=-1000.0, le=10000.0)
    unit: str = Field(..., max_length=20)
    quality: float = Field(default=1.0, ge=0.0, le=1.0)
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "sensor_id": "P001",
                "sensor_type": "pressure",
                "timestamp": "2025-11-13T22:00:00Z",
                "value": 150.5,
                "unit": "bar",
                "quality": 0.98
            }]
        }
    }


class Component(BaseModel):
    """Hydraulic system component."""
    
    id: str
    name: str
    component_type: ComponentType
    status: ComponentStatus = ComponentStatus.UNKNOWN
    parent_id: Optional[str] = None
    sensors: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    
    created_at: datetime
    updated_at: datetime


class HydraulicSystem(BaseModel):
    """Complete hydraulic system."""
    
    id: str
    name: str
    description: Optional[str] = None
    components: List[Component] = Field(default_factory=list)
    active: bool = True
    
    created_at: datetime
    updated_at: datetime
