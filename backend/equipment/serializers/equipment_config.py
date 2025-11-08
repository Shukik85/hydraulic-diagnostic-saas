"""Equipment Config Serializers"""
from rest_framework import serializers
from equipment.models.equipment_config import EquipmentConfig, ConfigExtractionLog


class EquipmentConfigSerializer(serializers.ModelSerializer):
    """Equipment configuration serializer"""
    
    uploaded_by_username = serializers.CharField(
        source="uploaded_by.username",
        read_only=True
    )
    approved_by_username = serializers.CharField(
        source="approved_by.username",
        read_only=True,
        allow_null=True
    )
    
    class Meta:
        model = EquipmentConfig
        fields = [
            "id",
            "equipment_id",
            "manufacturer",
            "model",
            "serial_number",
            "config_data",
            "status",
            "extraction_method",
            "uploaded_by_username",
            "approved_by_username",
            "approval_notes",
            "created_at",
            "updated_at",
            "approved_at",
            "deployed",
            "model_version",
        ]
        read_only_fields = [
            "id",
            "uploaded_by_username",
            "approved_by_username",
            "created_at",
            "updated_at",
        ]


class ManualUploadSerializer(serializers.Serializer):
    """Manual PDF upload serializer"""
    
    equipment_id = serializers.CharField(max_length=100)
    manufacturer = serializers.CharField(max_length=100)
    model = serializers.CharField(max_length=100)
    serial_number = serializers.CharField(max_length=100, required=False, allow_blank=True)
    manual_pdf = serializers.FileField()
    
    def validate_manual_pdf(self, value):
        """Validate PDF file"""
        if not value.name.endswith('.pdf'):
            raise serializers.ValidationError("File must be a PDF")
        
        # Max 50MB
        if value.size > 50 * 1024 * 1024:
            raise serializers.ValidationError("PDF file too large (max 50MB)")
        
        return value


class ConfigApprovalSerializer(serializers.Serializer):
    """Config approval serializer"""
    
    approved = serializers.BooleanField()
    notes = serializers.CharField(required=False, allow_blank=True)
