"""
Equipment metadata CRUD API
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from uuid import UUID
import structlog

from ..db.session import get_db
from ..models.equipment import Equipment, Component
from ..schemas.equipment import EquipmentCreate, EquipmentUpdate, EquipmentResponse
from ..middleware.auth import get_current_user
from ..models.user import User

router = APIRouter()
logger = structlog.get_logger()


@router.post("/", response_model=EquipmentResponse, status_code=status.HTTP_201_CREATED)
async def create_equipment(
    equipment: EquipmentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Create new equipment with components"""
    try:
        # Create equipment
        db_equipment = Equipment(
            user_id=current_user.id,
            system_id=equipment.system_id,
            system_type=equipment.system_type,
            name=equipment.name,
            description=equipment.description,
            adjacency_matrix=equipment.adjacency_matrix,
            location=equipment.location,
            manufacturer=equipment.manufacturer,
            model=equipment.model,
            serial_number=equipment.serial_number
        )

        # Create components
        for comp in equipment.components:
            db_component = Component(
                component_id=comp.component_id,
                component_type=comp.component_type,
                name=comp.name,
                sensors=comp.sensors,
                normal_ranges=comp.normal_ranges,
                connected_to=comp.connected_to,
                position=comp.position
            )
            db_equipment.components.append(db_component)

        db.add(db_equipment)
        await db.commit()
        await db.refresh(db_equipment)

        logger.info("equipment_created", equipment_id=str(db_equipment.id), user_id=str(current_user.id))
        return db_equipment

    except Exception as e:
        await db.rollback()
        logger.error("equipment_creation_failed", exc_info=e)
        raise HTTPException(status_code=500, detail="Failed to create equipment")


@router.get("/{system_id}", response_model=EquipmentResponse)
async def get_equipment(
    system_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get equipment by system_id"""
    result = await db.execute(
        select(Equipment).where(
            Equipment.user_id == current_user.id,
            Equipment.system_id == system_id
        )
    )
    equipment = result.scalar_one_or_none()

    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")

    return equipment


@router.get("/", response_model=List[EquipmentResponse])
async def list_equipment(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    """List all equipment for current user"""
    result = await db.execute(
        select(Equipment)
        .where(Equipment.user_id == current_user.id)
        .offset(skip)
        .limit(limit)
    )
    return result.scalars().all()


@router.patch("/{system_id}", response_model=EquipmentResponse)
async def update_equipment(
    system_id: str,
    equipment_update: EquipmentUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Update equipment"""
    result = await db.execute(
        select(Equipment).where(
            Equipment.user_id == current_user.id,
            Equipment.system_id == system_id
        )
    )
    equipment = result.scalar_one_or_none()

    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")

    # Update fields
    update_data = equipment_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(equipment, field, value)

    await db.commit()
    await db.refresh(equipment)

    logger.info("equipment_updated", equipment_id=str(equipment.id))
    return equipment


@router.delete("/{system_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_equipment(
    system_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete equipment"""
    result = await db.execute(
        select(Equipment).where(
            Equipment.user_id == current_user.id,
            Equipment.system_id == system_id
        )
    )
    equipment = result.scalar_one_or_none()

    if not equipment:
        raise HTTPException(status_code=404, detail="Equipment not found")

    await db.delete(equipment)
    await db.commit()

    logger.info("equipment_deleted", equipment_id=str(equipment.id))
