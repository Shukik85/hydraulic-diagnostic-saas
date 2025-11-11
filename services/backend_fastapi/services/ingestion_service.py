"""
Sensor data ingestion and validation service
"""

import uuid
from typing import Any

import structlog
from models.equipment import Equipment
from models.sensor_data import SensorData
from schemas.sensor import SensorReading
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


class IngestionService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def ingest_batch(
        self, user_id: uuid.UUID, system_id: str, readings: list[SensorReading]
    ) -> dict[str, Any]:
        """
        Ingest batch of sensor readings with validation
        """
        # Fetch equipment metadata for validation
        result = await self.db.execute(
            select(Equipment).where(
                and_(Equipment.user_id == user_id, Equipment.system_id == system_id)
            )
        )
        equipment = result.scalar_one_or_none()

        if not equipment:
            raise ValueError(f"Equipment {system_id} not found")

        # Build component lookup
        component_map = {c.component_id: c for c in equipment.components}

        ingested_count = 0
        quarantined_count = 0
        errors = []
        ingestion_id = uuid.uuid4()

        # Validate and insert readings
        for reading in readings:
            is_valid, validation_errors = self._validate_reading(reading, component_map)

            sensor_data = SensorData(
                user_id=user_id,
                system_id=system_id,
                component_id=reading.component_id,
                sensor_name=reading.sensor_name,
                value=reading.value,
                unit=reading.unit,
                timestamp=reading.timestamp,
                is_valid=is_valid,
                is_quarantined=not is_valid,
                validation_errors=(validation_errors if not is_valid else None),
            )

            self.db.add(sensor_data)

            if is_valid:
                ingested_count += 1
            else:
                quarantined_count = None if is_valid else validation_errors
                errors.extend(validation_errors)

        await self.db.commit()

        return {
            "ingested_count": ingested_count,
            "quarantined_count": quarantined_count,
            "errors": errors,
            "ingestion_id": ingestion_id,
        }

    def _validate_reading(
        self, reading: SensorReading, component_map: dict[str, Any]
    ) -> tuple[bool, list[str]]:
        """Validate sensor reading against metadata"""
        errors = []

        # Check if component exists
        if reading.component_id not in component_map:
            errors.append(f"Unknown component: {reading.component_id}")
            return False, errors

        component = component_map[reading.component_id]

        # Check if sensor exists for component
        if reading.sensor_name not in component.sensors:
            errors.append(
                f"Sensor {reading.sensor_name} not defined for {reading.component_id}"
            )
            return False, errors

        # Check value against normal ranges
        if reading.sensor_name in component.normal_ranges:
            ranges = component.normal_ranges[reading.sensor_name]
            if "min" in ranges and reading.value < ranges["min"]:
                errors.append(
                    f"{reading.sensor_name} below minimum: {reading.value} < {ranges['min']}"
                )
            if "max" in ranges and reading.value > ranges["max"]:
                errors.append(
                    f"{reading.sensor_name} above maximum: {reading.value} > {ranges['max']}"
                )

        return not errors, errors

    async def process_staging_to_hypertable(self, ingestion_id: uuid.UUID):
        """Move validated data from staging to TimescaleDB hypertable"""
        # TODO: Implement batch insert into hypertable
        logger.info("processing_to_hypertable", ingestion_id=str(ingestion_id))

    async def get_latest_readings(
        self, user_id: uuid.UUID, system_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get latest sensor readings for system"""
        result = await self.db.execute(
            select(SensorData)
            .where(
                and_(
                    SensorData.user_id == user_id,
                    SensorData.system_id == system_id,
                    SensorData.is_valid,
                )
            )
            .order_by(SensorData.timestamp.desc())
            .limit(limit)
        )

        readings = result.scalars().all()
        return [
            {
                "component_id": r.component_id,
                "sensor_name": r.sensor_name,
                "value": r.value,
                "unit": r.unit,
                "timestamp": r.timestamp,
            }
            for r in readings
        ]
