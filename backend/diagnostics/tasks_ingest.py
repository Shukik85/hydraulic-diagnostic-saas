from celery import shared_task
from diagnostics.models import SensorData, HydraulicSystem
from uuid import UUID
from django.utils import timezone
import logging

logger = logging.getLogger(__name__)

@shared_task
def ingest_sensor_data_bulk(system_id, readings, job_id=None):
    try:
        system = HydraulicSystem.objects.get(id=UUID(system_id))
    except HydraulicSystem.DoesNotExist:
        logger.error(f"System {system_id} not found for ingestion job {job_id}")
        # Optionally update job status in cache/Redis: failed
        return {"job_id": job_id, "status": "failed", "error": "system not found"}
    objs = []
    now = timezone.now()
    for item in readings:
        try:
            objs.append(SensorData(
                system=system,
                timestamp=item['timestamp'],
                sensor_type=item.get('quality', 'unknown'),
                value=item['value'],
                unit=item['unit'],
                # другие поля по необходимости
            ))
        except Exception as ex:
            logger.warning(f"Bad reading in job {job_id}: {item} ({ex})")
            continue
    SensorData.objects.bulk_create(objs, batch_size=1000)
    # Optionally update job status to finished
    return {"job_id": job_id, "status": "done", "count": len(objs)}
