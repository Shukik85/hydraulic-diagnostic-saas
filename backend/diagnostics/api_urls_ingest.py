from django.urls import path
from diagnostics.api_ingest import SensorBulkIngestAPIView

urlpatterns = [
    path('data/ingest', SensorBulkIngestAPIView.as_view(), name='api-sensor-bulk-ingest'),
]
