"""
URL routing for ingestion and job tracking API.

OpenAPI v3.1 compliant endpoints.
"""

from django.urls import path
from diagnostics.api_ingest import SensorBulkIngestAPIView
from diagnostics.api_job_status import IngestionJobStatusAPIView

urlpatterns = [
    # POST /api/v1/data/ingest - Bulk sensor data ingestion
    path('data/ingest', SensorBulkIngestAPIView.as_view(), name='api-sensor-bulk-ingest'),
    
    # GET /api/v1/jobs/{job_id}/ - Get ingestion job status
    path('jobs/<str:job_id>/', IngestionJobStatusAPIView.as_view(), name='api-job-status'),
]
