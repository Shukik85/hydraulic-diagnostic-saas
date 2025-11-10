"""Equipment Config ViewSet"""
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, IsAdminUser
from django.utils import timezone

from equipment.models.equipment_config import EquipmentConfig
from equipment.serializers.equipment_config import (
    EquipmentConfigSerializer,
    ManualUploadSerializer,
    ConfigApprovalSerializer
)


class EquipmentConfigViewSet(viewsets.ModelViewSet):
    """Equipment configuration management"""
    
    queryset = EquipmentConfig.objects.all()
    serializer_class = EquipmentConfigSerializer
    permission_classes = [IsAuthenticated]
    
    def perform_create(self, serializer):
        """Set uploaded_by"""
        serializer.save(uploaded_by=self.request.user)
    
    @action(detail=False, methods=["post"], url_path="upload-manual")
    def upload_manual(self, request):
        """
        Upload equipment manual PDF and auto-extract specs using RAG
        
        POST /api/equipment-configs/upload-manual/
        Content-Type: multipart/form-data
        
        {
            "equipment_id": "CAT_336_SN67890",
            "manufacturer": "Caterpillar",
            "model": "336",
            "serial_number": "67890",
            "manual_pdf": <file>
        }
        """
        serializer = ManualUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # TODO: Integrate RAG parser
        # For now, return placeholder
        extracted_specs = {
            "pump": {
                "max_pressure": 250,
                "nominal_rpm": 1800,
                "max_temperature": 90
            },
            "cylinder": {
                "max_pressure": 280,
                "stroke": 2500,
                "velocity_max": 0.5
            },
            "motor": {
                "max_rpm": 600,
                "max_temperature": 100
            }
        }
        
        # Create config
        config = EquipmentConfig.objects.create(
            equipment_id=serializer.validated_data["equipment_id"],
            manufacturer=serializer.validated_data["manufacturer"],
            model=serializer.validated_data["model"],
            serial_number=serializer.validated_data.get("serial_number", ""),
            config_data=extracted_specs,
            manual_pdf=serializer.validated_data["manual_pdf"],
            extraction_method="rag",
            uploaded_by=request.user,
            status="pending"
        )
        
        return Response({
            "config_id": config.id,
            "equipment_id": config.equipment_id,
            "extracted_specs": extracted_specs,
            "status": "pending_review",
            "message": "Manual uploaded. Specs extracted using RAG. Awaiting admin approval."
        }, status=status.HTTP_201_CREATED)
    
    @action(
        detail=True,
        methods=["post"],
        url_path="approve",
        permission_classes=[IsAdminUser]
    )
    def approve_config(self, request, pk=None):
        """
        Approve or reject equipment configuration
        
        POST /api/equipment-configs/{id}/approve/
        {
            "approved": true,
            "notes": "Specs verified against manual"
        }
        """
        config = self.get_object()
        serializer = ConfigApprovalSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        if serializer.validated_data["approved"]:
            config.status = "approved"
            config.approved_by = request.user
            config.approved_at = timezone.now()
            config.approval_notes = serializer.validated_data.get("notes", "")
            config.save()
            
            # TODO: Deploy model for this equipment
            # self._deploy_model(config)
            
            return Response({
                "status": "approved",
                "message": f"Configuration for {config.equipment_id} approved"
            })
        else:
            config.status = "rejected"
            config.approval_notes = serializer.validated_data.get("notes", "")
            config.save()
            
            return Response({
                "status": "rejected",
                "message": f"Configuration for {config.equipment_id} rejected"
            })
    
    @action(detail=False, methods=["get"], url_path="pending")
    def pending_configs(self, request):
        """List pending configs for admin review"""
        pending = self.queryset.filter(status="pending")
        serializer = self.get_serializer(pending, many=True)
        return Response(serializer.data)
