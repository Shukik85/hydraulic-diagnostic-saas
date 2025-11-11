"""
Sensor data ingestion API
High-performance bulk ingestion with validation
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
import structlog
import uuid

from ..db.session import get_db
from ..schemas.sensor import SensorDataIngest, SensorDataResponse
from ..services.ingestion_service import IngestionService
from ..middleware.auth import get_current_user
from ..models.user import User

router = APIRouter()
logger = structlog.get_logger()


@router.post("/ingest", response_model=SensorDataResponse)
async def ingest_sensor_data(
    data: SensorDataIngest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Ingest sensor data in batches
    Validates against equipment metadata and quarantines invalid readings
    """
    try:
        ingestion_service = IngestionService(db)

        result = await ingestion_service.ingest_batch(
            user_id=current_user.id,
            system_id=data.system_id,
            readings=data.readings
        )

        # Background task: Move validated data to TimescaleDB hypertable
        if result["ingested_count"] > 0:
            background_tasks.add_task(
                ingestion_service.process_staging_to_hypertable,
                ingestion_id=result["ingestion_id"]
            )

        logger.info(
            "sensor_data_ingested",
            user_id=str(current_user.id),
            system_id=data.system_id,
            count=result["ingested_count"],
            quarantined=result["quarantined_count"]
        )

        return result

    except Exception as e:
        logger.error("ingestion_failed", exc_info=e, user_id=str(current_user.id))
        raise HTTPException(status_code=500, detail="Ingestion failed")


@router.get("/systems/{system_id}/latest")
async def get_latest_readings(
    system_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    limit: int = 100
):
    """Get latest sensor readings for system"""
    ingestion_service = IngestionService(db)
    readings = await ingestion_service.get_latest_readings(
        user_id=current_user.id,
        system_id=system_id,
        limit=limit
    )
    return readings
