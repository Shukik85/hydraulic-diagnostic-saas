"""
Equipment models (read-only, mirrored from FastAPI DB)
"""
from django.db import models
import uuid

class Equipment(models.Model):
    """Equipment metadata (managed by FastAPI)"""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user_id = models.UUIDField(db_index=True)
    system_id = models.CharField(max_length=255, unique=True)
    system_type = models.CharField(max_length=100)
    name = models.CharField(max_length=255)

    adjacency_matrix = models.JSONField()
    components = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'equipment'
        managed = False  # Django won't create/modify this table
        verbose_name = 'Equipment'
        verbose_name_plural = 'Equipment'

    def __str__(self):
        return f"{self.system_id} - {self.name}"
