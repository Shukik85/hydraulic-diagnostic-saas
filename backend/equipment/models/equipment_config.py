"""Equipment Configuration Model"""
from django.db import models
from django.contrib.auth.models import User


class EquipmentConfig(models.Model):
    """Equipment configuration with physics-informed thresholds"""
    
    # Basic info
    equipment_id = models.CharField(max_length=100, unique=True, db_index=True)
    manufacturer = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    serial_number = models.CharField(max_length=100, blank=True)
    
    # Configuration JSON
    config_data = models.JSONField(
        help_text="Physics-informed thresholds and specifications"
    )
    
    # Source tracking
    manual_pdf = models.FileField(
        upload_to="equipment_manuals/",
        null=True,
        blank=True,
        help_text="Original equipment manual PDF"
    )
    extraction_method = models.CharField(
        max_length=50,
        choices=[
            ("manual", "Manual Entry"),
            ("rag", "RAG Auto-extraction"),
            ("api", "API Import")
        ],
        default="rag"
    )
    
    # Approval workflow
    status = models.CharField(
        max_length=20,
        choices=[
            ("pending", "Pending Review"),
            ("approved", "Approved"),
            ("rejected", "Rejected"),
        ],
        default="pending",
        db_index=True
    )
    
    uploaded_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        related_name="uploaded_configs"
    )
    
    approved_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="approved_configs"
    )
    
    approval_notes = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    
    # Deployment
    deployed = models.BooleanField(default=False)
    deployed_at = models.DateTimeField(null=True, blank=True)
    model_version = models.CharField(max_length=50, blank=True)
    
    class Meta:
        db_table = "equipment_configs"
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["equipment_id", "status"]),
            models.Index(fields=["manufacturer", "model"]),
        ]
    
    def __str__(self):
        return f"{self.equipment_id} - {self.manufacturer} {self.model}"
    
    @property
    def is_approved(self):
        return self.status == "approved"


class ConfigExtractionLog(models.Model):
    """Log of RAG extraction attempts"""
    
    config = models.ForeignKey(
        EquipmentConfig,
        on_delete=models.CASCADE,
        related_name="extraction_logs"
    )
    
    extracted_data = models.JSONField()
    confidence_scores = models.JSONField(null=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "config_extraction_logs"
        ordering = ["-created_at"]
