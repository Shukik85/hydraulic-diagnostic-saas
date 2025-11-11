"""
Sensor Mapping API
Level 6A: Assign sensors to components
"""

import uuid

from db.session import get_db
from fastapi import APIRouter, Depends, HTTPException, status
from middleware.auth import get_current_user
from models.equipment import Equipment
from models.sensor_mapping import SensorMapping
from schemas.sensor_mapping import (
    AutoDetectResponse,
    SensorMappingCreate,
    SensorMappingResponse,
    SensorMappingUpdate,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/api/sensor-mappings", tags=["Sensor Mapping"])


@router.post(
    "/", response_model=SensorMappingResponse, status_code=status.HTTP_201_CREATED
)
async def create_sensor_mapping(
    mapping: SensorMappingCreate,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """
    Create new sensor-to-component mapping
    """
    # Verify equipment exists and belongs to user
    equipment = await db.get(Equipment, mapping.equipment_id)
    if not equipment or equipment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Equipment not found")

    # Verify component_index is valid
    if mapping.component_index >= len(equipment.components):
        raise HTTPException(status_code=400, detail="Invalid component_index")

    # Check sensor_id uniqueness
    existing = await db.execute(
        select(SensorMapping).where(SensorMapping.sensor_id == mapping.sensor_id)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Sensor ID already mapped")

    # Create mapping
    db_mapping = SensorMapping(
        **mapping.dict(), component_name=equipment.components[mapping.component_index]
    )

    db.add(db_mapping)
    await db.commit()
    await db.refresh(db_mapping)

    return db_mapping


@router.get("/equipment/{equipment_id}", response_model=list[SensorMappingResponse])
async def get_equipment_sensors(
    equipment_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """
    Get all sensor mappings for equipment
    """
    equipment = await db.get(Equipment, equipment_id)
    if not equipment or equipment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Equipment not found")

    result = await db.execute(
        select(SensorMapping).where(SensorMapping.equipment_id == equipment_id)
    )
    return result.scalars().all()


@router.post("/equipment/{equipment_id}/auto-detect", response_model=AutoDetectResponse)
async def auto_detect_sensors(
    equipment_id: uuid.UUID,
    available_sensors: list[str],  # List of sensor IDs from data source
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """
    Auto-detect sensor-to-component mappings
    Uses naming conventions and heuristics
    """
    equipment = await db.get(Equipment, equipment_id)
    if not equipment or equipment.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Equipment not found")

    suggestions = []

    for component_idx, component_name in enumerate(equipment.components):
        # Find sensors matching component name
        matching_sensors = [
            s for s in available_sensors if component_name.lower() in s.lower()
        ]

        for sensor_id in matching_sensors:
            # Infer sensor type from ID
            sensor_type = infer_sensor_type(sensor_id)

            # Estimate expected range based on component type
            ranges = estimate_ranges(
                component_type=equipment.component_specs.get(component_name, {}).get(
                    "type"
                ),
                sensor_type=sensor_type,
            )

            suggestions.append(
                {
                    "component_index": component_idx,
                    "component_name": component_name,
                    "sensor_id": sensor_id,
                    "sensor_type": sensor_type,
                    "expected_range_min": ranges["min"],
                    "expected_range_max": ranges["max"],
                    "unit": ranges["unit"],
                    "confidence": calculate_confidence(sensor_id, component_name),
                    "auto_detected": True,
                }
            )

    return {
        "equipment_id": equipment_id,
        "total_sensors": len(available_sensors),
        "matched_sensors": len(suggestions),
        "suggestions": suggestions,
    }


@router.patch("/{mapping_id}", response_model=SensorMappingResponse)
async def update_sensor_mapping(
    mapping_id: uuid.UUID,
    update: SensorMappingUpdate,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """Update sensor mapping"""
    mapping = await db.get(SensorMapping, mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    # Verify ownership
    equipment = await db.get(Equipment, mapping.equipment_id)
    if equipment.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Update fields
    for field, value in update.dict(exclude_unset=True).items():
        setattr(mapping, field, value)

    await db.commit()
    await db.refresh(mapping)

    return mapping


@router.delete("/{mapping_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sensor_mapping(
    mapping_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),  # noqa: B008
    current_user=Depends(get_current_user),  # noqa: B008
):
    """Delete sensor mapping"""
    mapping = await db.get(SensorMapping, mapping_id)
    if not mapping:
        raise HTTPException(status_code=404, detail="Mapping not found")

    # Verify ownership
    equipment = await db.get(Equipment, mapping.equipment_id)
    if equipment.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")

    await db.delete(mapping)
    await db.commit()


# Helper functions
def infer_sensor_type(sensor_id: str) -> str:
    """Infer sensor type from ID"""
    sensor_id_lower = sensor_id.lower()

    if "p" in sensor_id_lower or "pressure" in sensor_id_lower:
        return "pressure"
    elif "t" in sensor_id_lower or "temp" in sensor_id_lower:
        return "temperature"
    elif "v" in sensor_id_lower or "vibr" in sensor_id_lower:
        return "vibration"
    elif "f" in sensor_id_lower or "flow" in sensor_id_lower:
        return "flow"
    else:
        return "unknown"


def estimate_ranges(component_type: str, sensor_type: str) -> dict:
    """Estimate expected ranges based on component and sensor type"""
    ranges_map = {
        "pump": {
            "pressure": {"min": 150, "max": 250, "unit": "bar"},
            "temperature": {"min": 40, "max": 80, "unit": "°C"},
            "vibration": {"min": 0, "max": 5, "unit": "mm/s"},
        },
        "cylinder": {
            "pressure": {"min": 100, "max": 200, "unit": "bar"},
            "temperature": {"min": 30, "max": 70, "unit": "°C"},
        },
        "valve": {"pressure": {"min": 50, "max": 150, "unit": "bar"}},
    }

    if component_type in ranges_map and sensor_type in ranges_map[component_type]:
        return ranges_map[component_type][sensor_type]

    default_ranges = {
        "pressure": {"min": 0, "max": 300, "unit": "bar"},
        "temperature": {"min": 0, "max": 100, "unit": "°C"},
        "vibration": {"min": 0, "max": 10, "unit": "mm/s"},
        "flow": {"min": 0, "max": 200, "unit": "L/min"},
    }

    return default_ranges.get(sensor_type, {"min": 0, "max": 100, "unit": "units"})


def calculate_confidence(sensor_id: str, component_name: str) -> float:
    """Calculate confidence score for auto-detection"""
    score = 0.0

    # Exact match in sensor ID
    if component_name.lower() in sensor_id.lower():
        score += 0.6

    # Sensor type inference
    sensor_type = infer_sensor_type(sensor_id)
    if sensor_type != "unknown":
        score += 0.3

    # Additional heuristics
    if sensor_id.startswith(component_name):
        score += 0.1

    return min(1.0, score)
