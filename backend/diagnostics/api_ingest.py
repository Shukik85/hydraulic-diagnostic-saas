"""
Sensor data bulk ingestion API endpoint.

Enterprise-grade ingestion with:
- JWT authentication
- Rate limiting
- Job tracking
- Async processing
"""

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from uuid import uuid4, UUID
from django.utils import timezone
import logging

from diagnostics.serializers_ingest import SensorBulkIngestSerializer
from diagnostics.models_ingestion import IngestionJob
from diagnostics.tasks_ingest import ingest_sensor_data_bulk

logger = logging.getLogger(__name__)


class BurstUserRateThrottle(UserRateThrottle):
    """Rate limiter: 15 requests per minute per user."""
    rate = '15/min'


class SensorBulkIngestAPIView(APIView):
    """
    POST /api/v1/data/ingest endpoint.
    
    Accepts bulk sensor readings and queues them for async processing.
    Complies with OpenAPI v3.1 specification.
    
    Features:
    - JWT authentication required
    - Rate limiting (15 req/min per user)
    - Validation (1-10,000 readings per batch)
    - Async processing via Celery
    - Job tracking for observability
    
    Returns:
        202 Accepted: {job_id, status}
        400 Bad Request: Validation error
        401 Unauthorized: Authentication required
        429 Too Many Requests: Rate limit exceeded
    """
    
    permission_classes = [IsAuthenticated]
    throttle_classes = [BurstUserRateThrottle]

    def post(self, request, *args, **kwargs):
        """
        Handle bulk sensor data ingestion.
        
        Request body:
            {
                "system_id": "uuid",
                "readings": [{sensor_id, timestamp, value, unit, quality}]
            }
        
        Returns:
            202: {job_id, status: "queued"}
        """
        # Validate request data
        serializer = SensorBulkIngestSerializer(data=request.data)
        
        try:
            serializer.is_valid(raise_exception=True)
        except Exception as e:
            logger.warning(f"Validation error in bulk ingest: {e}")
            return Response(
                {
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid request data",
                        "details": serializer.errors if hasattr(serializer, 'errors') else {},
                        "timestamp": timezone.now().isoformat(),
                    }
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        # Extract validated data
        system_id = serializer.validated_data["system_id"]
        readings = serializer.validated_data["readings"]
        
        # Generate job ID
        job_id = uuid4()
        
        # Create IngestionJob record
        try:
            job = IngestionJob.objects.create(
                id=job_id,
                system_id=UUID(str(system_id)),
                status='queued',
                total_readings=len(readings),
                created_by=request.user,
            )
            
            logger.info(
                f"Created ingestion job {job_id} for system {system_id} "
                f"with {len(readings)} readings by user {request.user.username}"
            )
            
        except Exception as e:
            logger.error(f"Failed to create ingestion job: {e}")
            return Response(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to create ingestion job",
                        "details": {"error": str(e)},
                        "timestamp": timezone.now().isoformat(),
                    }
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
        # Queue Celery task for async processing
        try:
            # Convert readings to dict format for Celery
            readings_data = [
                {
                    'sensor_id': str(r['sensor_id']),
                    'timestamp': r['timestamp'].isoformat() if hasattr(r['timestamp'], 'isoformat') else str(r['timestamp']),
                    'value': float(r['value']),
                    'unit': r['unit'],
                    'quality': int(r.get('quality', 100)),
                }
                for r in readings
            ]
            
            celery_result = ingest_sensor_data_bulk.delay(
                system_id=str(system_id),
                readings=readings_data,
                job_id=str(job_id),
                user_id=str(request.user.id) if request.user else None,
            )
            
            # Update job with Celery task ID
            job.celery_task_id = celery_result.id
            job.save(update_fields=['celery_task_id'])
            
            logger.info(f"Queued Celery task {celery_result.id} for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to queue Celery task for job {job_id}: {e}")
            
            # Mark job as failed
            job.status = 'failed'
            job.error_message = f"Failed to queue Celery task: {e}"
            job.completed_at = timezone.now()
            job.save()
            
            return Response(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Failed to queue ingestion task",
                        "details": {"error": str(e)},
                        "timestamp": timezone.now().isoformat(),
                    }
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
        # Return 202 Accepted with job_id
        return Response(
            {
                "job_id": str(job_id),
                "status": "queued",
            },
            status=status.HTTP_202_ACCEPTED,
        )


__all__ = [
    'SensorBulkIngestAPIView',
    'BurstUserRateThrottle',
]
