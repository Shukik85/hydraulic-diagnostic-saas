"""
Equipment Configuration API
Manages physics-informed equipment metadata
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.core.cache import cache
from django.contrib.auth.decorators import permission_required
from .models import EquipmentConfig as EquipmentConfigModel
from .serializers import EquipmentConfigSerializer
import json


class EquipmentConfigView(APIView):
    """
    POST /api/equipment/config/
    Upload or update equipment configuration
    """
    
    def post(self, request):
        customer_id = request.user.customer_id
        equipment_id = request.data.get("equipment_id")
        
        # Option 1: User uploads config manually
        if "config_json" in request.data:
            config_json = request.data["config_json"]
        
        # Option 2: User uploads PDF manual → RAG extraction
        elif "manual_pdf" in request.FILES:
            pdf_file = request.FILES["manual_pdf"]
            
            # RAG extraction (async task)
            from backend.tasks.rag_extraction import extract_equipment_specs
            task = extract_equipment_specs.delay(
                pdf_path=pdf_file.path,
                equipment_model=request.data.get("model")
            )
            
            return Response({
                "status": "processing",
                "task_id": task.id,
                "message": "Extracting specs from manual..."
            })
        else:
            return Response(
                {"error": "Provide either config_json or manual_pdf"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate config structure
        try:
            # Basic validation
            assert "components" in config_json
            assert "pump" in config_json["components"]
        except (AssertionError, KeyError) as e:
            return Response(
                {"error": f"Invalid config structure: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Save to database
        config_obj, created = EquipmentConfigModel.objects.update_or_create(
            equipment_id=equipment_id,
            customer_id=customer_id,
            defaults={
                "config_json": config_json,
                "equipment_model": config_json.get("model"),
                "manufacturer": config_json.get("manufacturer"),
                "approved": False  # Requires admin approval
            }
        )
        
        # Invalidate cache
        cache.delete(f"equipment_config:{equipment_id}")
        
        return Response({
            "status": "pending_approval" if not config_obj.approved else "active",
            "equipment_id": equipment_id,
            "config_id": str(config_obj.id),
            "message": "Config submitted for review" if not created else "Config updated"
        }, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)
    
    def get(self, request, equipment_id=None):
        """GET /api/equipment/config/<equipment_id>/"""
        if not equipment_id:
            return Response({"error": "equipment_id required"}, status=400)
        
        try:
            config = EquipmentConfigModel.objects.get(
                equipment_id=equipment_id,
                customer_id=request.user.customer_id
            )
            serializer = EquipmentConfigSerializer(config)
            return Response(serializer.data)
        except EquipmentConfigModel.DoesNotExist:
            return Response(
                {"error": "Equipment config not found"},
                status=status.HTTP_404_NOT_FOUND
            )


class SensorInferenceView(APIView):
    """
    POST /api/inference/
    Real-time anomaly detection with equipment-aware normalization
    """
    
    def post(self, request):
        sensor_data = request.data.get("sensor_data")
        equipment_id = request.data.get("equipment_id")
        component_type = request.data.get("component_type")
        
        if not all([sensor_data, equipment_id, component_type]):
            return Response(
                {"error": "Missing required fields"},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Load equipment config (cached)
        config = self._get_equipment_config(equipment_id)
        if not config:
            return Response(
                {"error": "Equipment config not found or not approved"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Trigger ML inference (async)
        from backend.tasks.inference import predict_anomaly
        task = predict_anomaly.delay(
            sensor_data=sensor_data,
            equipment_id=equipment_id,
            component_type=component_type,
            config_json=config
        )
        
        return Response({
            "task_id": task.id,
            "status": "processing",
            "equipment_id": equipment_id,
            "component_type": component_type
        })
    
    def get(self, request, task_id):
        """GET /api/inference/result/<task_id>/"""
        from celery.result import AsyncResult
        
        result = AsyncResult(task_id)
        
        if result.ready():
            return Response({
                "status": "completed",
                "result": result.get()
            })
        else:
            return Response({
                "status": "processing",
                "task_id": task_id
            })
    
    def _get_equipment_config(self, equipment_id):
        """Load config from cache or DB"""
        cache_key = f"equipment_config:{equipment_id}"
        config_json = cache.get(cache_key)
        
        if not config_json:
            try:
                db_config = EquipmentConfigModel.objects.get(
                    equipment_id=equipment_id,
                    approved=True
                )
                config_json = db_config.config_json
                
                # Cache for 1 hour
                cache.set(cache_key, config_json, timeout=3600)
            except EquipmentConfigModel.DoesNotExist:
                return None
        
        return config_json
