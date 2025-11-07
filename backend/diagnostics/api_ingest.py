from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.throttling import UserRateThrottle
from diagnostics.serializers_ingest import SensorBulkIngestSerializer
from uuid import uuid4
from celery.result import AsyncResult
from django.core.cache import cache
from diagnostics.tasks import ingest_sensor_data_bulk

class BurstUserRateThrottle(UserRateThrottle):
    rate = '15/min'

class SensorBulkIngestAPIView(APIView):
    permission_classes = [IsAuthenticated]
    throttle_classes = [BurstUserRateThrottle]

    def post(self, request, *args, **kwargs):
        serializer = SensorBulkIngestSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        system_id = serializer.validated_data["system_id"]
        readings = serializer.validated_data["readings"]
        job_id = str(uuid4())
        # Запуск фоновой задачи ingestion на Celery
        celery_result = ingest_sensor_data_bulk.delay(system_id, readings, job_id)
        # Можно сохранять статус в Redis
        cache.set(f"ingest_job_status:{job_id}", "queued", timeout=900)
        return Response({"job_id": job_id, "status": "queued"}, status=status.HTTP_202_ACCEPTED)
