"""
Celery tasks for sensor data bulk ingestion.

Enterprise-grade ingestion pipeline with:
- Validation and quarantine logic
- Job status tracking
- Performance optimization
- Full error handling and logging
"""

import logging
import time
from typing import Any
from uuid import UUID

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from datetime import timedelta

from diagnostics.models import SensorData, HydraulicSystem
from diagnostics.models_ingestion import IngestionJob
from diagnostics.models_quarantine import QuarantinedReading

logger = logging.getLogger(__name__)

# FIXED: Import validation config from correct location
try:
    from config.sensor_validation import (  # CHANGED: project.settings -> config
        SENSOR_VALUE_RANGES,
        QUALITY_THRESHOLDS,
        TIMESTAMP_VALIDATION,
    )
except ImportError:
    # Fallback defaults if config not found
    logger.warning("sensor_validation config not found, using defaults")
    SENSOR_VALUE_RANGES = {}
    QUALITY_THRESHOLDS = {"poor": 50}
    TIMESTAMP_VALIDATION = {"max_future_offset": 300, "max_past_offset": 157680000}


def validate_reading(
    reading: dict[str, Any], system: HydraulicSystem
) -> tuple[bool, str, str]:
    """
    Validate a single sensor reading.
    
    Returns:
        (is_valid, reason, reason_details)
    """
    # Extract fields
    value = reading.get('value')
    unit = reading.get('unit')
    timestamp = reading.get('timestamp')
    quality = reading.get('quality', 100)
    
    # Validate timestamp
    now = timezone.now()
    max_future = now + timedelta(seconds=TIMESTAMP_VALIDATION['max_future_offset'])
    max_past = now - timedelta(seconds=TIMESTAMP_VALIDATION['max_past_offset'])
    
    if timestamp > max_future:
        return False, 'invalid_timestamp', f"Timestamp {timestamp} is in the future (max: {max_future})"
    
    if timestamp < max_past:
        return False, 'invalid_timestamp', f"Timestamp {timestamp} is too old (min: {max_past})"
    
    # Validate quality score
    if quality < QUALITY_THRESHOLDS.get('poor', 50):
        return False, 'out_of_range', f"Quality score {quality} below threshold"
    
    # Validate unit
    if unit not in SENSOR_VALUE_RANGES:
        return False, 'invalid_unit', f"Unknown unit: {unit}"
    
    # Validate value range
    ranges = SENSOR_VALUE_RANGES.get(unit, {})
    min_val = ranges.get('min')
    max_val = ranges.get('max')
    
    if min_val is not None and value < min_val:
        return False, 'out_of_range', f"Value {value} < min ({min_val}) for unit {unit}"
    
    if max_val is not None and value > max_val:
        return False, 'out_of_range', f"Value {value} > max ({max_val}) for unit {unit}"
    
    return True, '', ''


def chunked_bulk_create(model, objects: list, batch_size: int = 1000) -> int:
    """
    Bulk create objects in chunks for optimal performance.
    
    Returns:
        Total number of objects created
    """
    total_created = 0
    for i in range(0, len(objects), batch_size):
        chunk = objects[i:i + batch_size]
        model.objects.bulk_create(chunk, batch_size=batch_size)
        total_created += len(chunk)
    return total_created


@shared_task(bind=True, max_retries=3)
def ingest_sensor_data_bulk(
    self,
    system_id: str,
    readings: list[dict[str, Any]],
    job_id: str,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Bulk ingest sensor data with validation and quarantine.
    
    Args:
        system_id: Target hydraulic system UUID
        readings: List of sensor readings
        job_id: Ingestion job UUID
        user_id: User who initiated ingestion (optional)
    
    Returns:
        Job status dictionary
    """
    start_time = time.time()
    job_uuid = UUID(job_id)
    
    # Get or create job record
    try:
        job = IngestionJob.objects.get(id=job_uuid)
    except IngestionJob.DoesNotExist:
        logger.error(f"Job {job_id} not found - creating new record")
        job = IngestionJob.objects.create(
            id=job_uuid,
            system_id=UUID(system_id),
            status='queued',
            total_readings=len(readings),
            celery_task_id=self.request.id,
        )
    
    # Update job status to processing
    job.status = 'processing'
    job.started_at = timezone.now()
    job.celery_task_id = self.request.id
    job.save(update_fields=['status', 'started_at', 'celery_task_id'])
    
    try:
        # Verify system exists
        try:
            system = HydraulicSystem.objects.get(id=UUID(system_id))
        except HydraulicSystem.DoesNotExist:
            error_msg = f"System {system_id} not found"
            logger.error(f"{error_msg} for job {job_id}")
            
            # Quarantine all readings
            quarantined = []
            for reading in readings:
                quarantined.append(QuarantinedReading(
                    job_id=job_uuid,
                    sensor_id=UUID(reading.get('sensor_id', '00000000-0000-0000-0000-000000000000')),
                    timestamp=reading.get('timestamp', timezone.now()),
                    value=reading.get('value', 0.0),
                    unit=reading.get('unit', ''),
                    quality=reading.get('quality', 0),
                    system_id=UUID(system_id),
                    reason='system_not_found',
                    reason_details=error_msg,
                ))
            
            chunked_bulk_create(QuarantinedReading, quarantined)
            
            # Update job as failed
            job.status = 'failed'
            job.error_message = error_msg
            job.quarantined_readings = len(readings)
            job.completed_at = timezone.now()
            job.processing_time_ms = int((time.time() - start_time) * 1000)
            job.save()
            
            return {
                "job_id": job_id,
                "status": "failed",
                "error": error_msg,
                "quarantined": len(readings),
            }
        
        # Process readings
        valid_readings = []
        quarantined_readings = []
        
        for reading in readings:
            try:
                is_valid, reason, reason_details = validate_reading(reading, system)
                
                if is_valid:
                    # Add to valid readings
                    valid_readings.append(SensorData(
                        system=system,
                        timestamp=reading['timestamp'],
                        sensor_type=reading.get('sensor_type', 'pressure'),
                        value=reading['value'],
                        unit=reading['unit'],
                        is_critical=reading.get('quality', 100) < QUALITY_THRESHOLDS.get('acceptable', 70),
                    ))
                else:
                    # Quarantine invalid reading
                    quarantined_readings.append(QuarantinedReading(
                        job_id=job_uuid,
                        sensor_id=UUID(reading.get('sensor_id', '00000000-0000-0000-0000-000000000000')),
                        timestamp=reading['timestamp'],
                        value=reading['value'],
                        unit=reading['unit'],
                        quality=reading.get('quality', 0),
                        system_id=UUID(system_id),
                        reason=reason,
                        reason_details=reason_details,
                    ))
                    logger.warning(f"Quarantined reading in job {job_id}: {reason} - {reason_details}")
                    
            except Exception as e:
                # Parse error - quarantine
                logger.error(f"Failed to parse reading in job {job_id}: {e}")
                quarantined_readings.append(QuarantinedReading(
                    job_id=job_uuid,
                    sensor_id=UUID('00000000-0000-0000-0000-000000000000'),
                    timestamp=timezone.now(),
                    value=0.0,
                    unit='',
                    quality=0,
                    system_id=UUID(system_id),
                    reason='parse_error',
                    reason_details=str(e),
                ))
        
        # Bulk insert valid readings
        inserted_count = 0
        if valid_readings:
            with transaction.atomic():
                inserted_count = chunked_bulk_create(SensorData, valid_readings)
        
        # Bulk insert quarantined readings
        quarantined_count = 0
        if quarantined_readings:
            with transaction.atomic():
                quarantined_count = chunked_bulk_create(QuarantinedReading, quarantined_readings)
        
        # Update job as completed
        processing_time = int((time.time() - start_time) * 1000)
        job.status = 'completed'
        job.inserted_readings = inserted_count
        job.quarantined_readings = quarantined_count
        job.completed_at = timezone.now()
        job.processing_time_ms = processing_time
        job.save()
        
        logger.info(
            f"Job {job_id} completed: {inserted_count} inserted, "
            f"{quarantined_count} quarantined in {processing_time}ms"
        )
        
        return {
            "job_id": job_id,
            "status": "completed",
            "inserted": inserted_count,
            "quarantined": quarantined_count,
            "processing_time_ms": processing_time,
        }
        
    except Exception as e:
        # Unexpected error
        logger.exception(f"Unexpected error in job {job_id}: {e}")
        
        processing_time = int((time.time() - start_time) * 1000)
        job.status = 'failed'
        job.error_message = str(e)
        job.completed_at = timezone.now()
        job.processing_time_ms = processing_time
        job.save()
        
        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e, countdown=60)  # Retry after 1 minute
        
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "processing_time_ms": processing_time,
        }


__all__ = [
    'ingest_sensor_data_bulk',
    'validate_reading',
    'chunked_bulk_create',
]
