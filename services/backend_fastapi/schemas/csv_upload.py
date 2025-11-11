"""
Pydantic schemas for CSV upload
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid


class ValidationError(BaseModel):
    field: str
    message: str


class ValidationWarning(BaseModel):
    field: Optional[str] = None
    sensor: Optional[str] = None
    message: str
    severity: str = "medium"


class CSVStats(BaseModel):
    rows_total: int
    rows_valid: int
    rows_invalid: int
    sensors_count: int
    date_range: Dict[str, Any]


class CSVValidationResponse(BaseModel):
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationWarning]
    stats: Dict[str, Any]
    preview: List[Dict[str, Any]] = []


class CSVImportRequest(BaseModel):
    equipment_id: uuid.UUID
    ignore_warnings: bool = False


class CSVImportResponse(BaseModel):
    success: bool
    equipment_id: uuid.UUID
    data_source_id: uuid.UUID
    imported_readings: int
    date_range: Dict[str, Any]
