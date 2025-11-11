"""
Pydantic schemas for simulator
"""
from pydantic import BaseModel, Field
from typing import Literal, Dict, Any
from datetime import datetime
import uuid


class SimulatorConfig(BaseModel):
    equipment_id: uuid.UUID
    scenario: Literal["normal", "degradation", "failure", "cyclic"]
    duration: int = Field(ge=60, le=3600, default=300)  # seconds
    noise_level: float = Field(ge=0.0, le=1.0, default=0.1)
    sampling_rate: int = Field(ge=1, le=100, default=10)  # Hz


class SimulatorStartResponse(BaseModel):
    simulation_id: uuid.UUID
    data_source_id: uuid.UUID
    status: str
    estimated_readings: int


class SimulatorStatus(BaseModel):
    simulation_id: uuid.UUID
    status: str
    started_at: datetime
    config: Dict[str, Any]
