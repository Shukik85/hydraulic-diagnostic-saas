"""
Equipment-related Pydantic schemas
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime


class ComponentSchema(BaseModel):
    """Component definition within equipment"""
    component_id: str = Field(..., description="Unique component identifier")
    component_type: str = Field(..., description="Type: pump, valve, cylinder, motor")
    name: str = Field(..., description="Human-readable name")
    sensors: List[str] = Field(..., description="List of sensor types")
    normal_ranges: Dict[str, Dict[str, float]] = Field(
        ..., 
        description="Normal operating ranges per sensor",
        examples=[{"pressure": {"min": 10, "max": 250}}]
    )
    connected_to: Optional[List[str]] = Field(None, description="Connected component IDs")
    position: Optional[int] = Field(None, description="Visual position in UI")


class EquipmentCreate(BaseModel):
    """Request schema for creating new equipment"""
    system_id: str = Field(..., description="User-defined system ID")
    system_type: str = Field(..., description="press, excavator, crane, custom")
    name: str
    description: Optional[str] = None
    adjacency_matrix: List[List[int]] = Field(..., description="Component connectivity")
    components: List[ComponentSchema]

    # Optional metadata
    location: Optional[str] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    serial_number: Optional[str] = None

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "system_id": "press_01",
            "system_type": "press",
            "name": "Hydraulic Press #1",
            "adjacency_matrix": [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            "components": [
                {
                    "component_id": "pump",
                    "component_type": "pump",
                    "name": "Main Pump",
                    "sensors": ["pressure", "flow", "temperature"],
                    "normal_ranges": {
                        "pressure": {"min": 10, "max": 250},
                        "flow": {"min": 0, "max": 100}
                    },
                    "connected_to": ["valve_main"]
                }
            ]
        }
    })


class EquipmentUpdate(BaseModel):
    """Request schema for updating equipment"""
    name: Optional[str] = None
    description: Optional[str] = None
    adjacency_matrix: Optional[List[List[int]]] = None
    components: Optional[List[ComponentSchema]] = None
    is_active: Optional[bool] = None


class EquipmentResponse(BaseModel):
    """Response schema for equipment"""
    id: UUID
    user_id: UUID
    system_id: str
    system_type: str
    name: str
    description: Optional[str]
    adjacency_matrix: List[List[int]]
    components: List[ComponentSchema]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime]

    model_config = ConfigDict(from_attributes=True)
