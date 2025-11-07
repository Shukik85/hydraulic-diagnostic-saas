"""
API endpoint for ingestion job status tracking.

Provides real-time job status for bulk ingestion operations.
"""

from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.shortcuts import get_object_or_404
from uuid import UUID

from diagnostics.models_ingestion import IngestionJob


class IngestionJobStatusSerializer:
    """
    Serializer for IngestionJob status response.
    Matches OpenAPI IngestionJobStatus schema.
    """
    
    @staticmethod
    def serialize(job: IngestionJob) -> dict:
        """Serialize job to OpenAPI-compliant dict."""
        return {
            "job_id": str(job.id),
            "status": job.status,
            "total_readings": job.total_readings,
            "inserted_readings": job.inserted_readings,
            "quarantined_readings": job.quarantined_readings,
            "created_at": job.created_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message if job.error_message else None,
            "processing_time_ms": job.processing_time_ms,
            "success_rate": round(job.success_rate, 2) if job.total_readings > 0 else 0.0,
            "system_id": str(job.system_id),
        }


class IngestionJobStatusAPIView(APIView):
    """
    GET /api/v1/jobs/{job_id}/ endpoint.
    
    Returns detailed status of an ingestion job.
    Complies with OpenAPI v3.1 specification.
    """
    
    permission_classes = [IsAuthenticated]
    
    def get(self, request, job_id: str):
        """
        Retrieve ingestion job status.
        
        Args:
            job_id: UUID of the ingestion job
        
        Returns:
            200: Job status (IngestionJobStatus schema)
            404: Job not found
        """
        try:
            job_uuid = UUID(job_id)
        except ValueError:
            return Response(
                {
                    "error": {
                        "code": "VALIDATION_ERROR",
                        "message": "Invalid job_id format",
                        "details": {"job_id": "Must be a valid UUID"},
                        "timestamp": request.META.get('HTTP_X_REQUEST_ID', ''),
                    }
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        job = get_object_or_404(IngestionJob, id=job_uuid)
        
        # Verify user has access to this system's jobs
        # TODO: Add proper permission check based on system ownership
        
        response_data = IngestionJobStatusSerializer.serialize(job)
        
        return Response(response_data, status=status.HTTP_200_OK)


__all__ = [
    'IngestionJobStatusAPIView',
    'IngestionJobStatusSerializer',
]
