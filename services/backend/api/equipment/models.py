"""
Equipment Configuration Models
"""
from django.db import models
import uuid


class EquipmentConfig(models.Model):
    """Equipment configuration with physics-informed thresholds"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    customer_id = models.UUIDField(db_index=True)
    equipment_id = models.CharField(max_length=100, unique=True, db_index=True)
    equipment_model = models.CharField(max_length=100)
    manufacturer = models.CharField(max_length=100)
    config_json = models.JSONField()
    
    approved = models.BooleanField(default=False)
    approved_by = models.UUIDField(null=True, blank=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    
    version = models.IntegerField(default=1)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = "equipment_configs"
        indexes = [
            models.Index(fields=["customer_id", "equipment_id"]),
            models.Index(fields=["approved", "customer_id"])
        ]
    
    def __str__(self):
        return f"{self.equipment_id} ({self.equipment_model})"


class ModelDeployment(models.Model):
    """ML model deployment tracking"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    equipment = models.ForeignKey(
        EquipmentConfig,
        on_delete=models.CASCADE,
        related_name="deployments"
    )
    component_type = models.CharField(max_length=50)  # cylinder, pump, valve
    model_version = models.CharField(max_length=50)
    mlflow_run_id = models.CharField(max_length=100)
    
    accuracy = models.FloatField()
    f1_score = models.FloatField()
    
    status = models.CharField(
        max_length=20,
        choices=[
            ("active", "Active"),
            ("testing", "Testing"),
            ("deprecated", "Deprecated")
        ],
        default="active"
    )
    
    deployed_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = "model_deployments"
        indexes = [
            models.Index(fields=["equipment", "component_type", "status"])
        ]
