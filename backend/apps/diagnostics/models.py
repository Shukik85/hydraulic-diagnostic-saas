"""Optimized diagnostics models with full typing support (mypy-safe)."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

from django.contrib.postgres.indexes import BrinIndex, BTreeIndex
from django.core.exceptions import ValidationError
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.db.models import Q
from django.db.models.functions import TruncDay
from django.utils import timezone

if TYPE_CHECKING:
    from django.db.models import Manager as RelatedManager


class HydraulicSystemQuerySet(models.QuerySet["HydraulicSystem"]):
    def with_owner(self) -> "HydraulicSystemQuerySet":
        return self.select_related("owner")

    def with_components(self) -> "HydraulicSystemQuerySet":
        return self.prefetch_related("components")

    def active(self) -> "HydraulicSystemQuerySet":
        return self.filter(status="active")

    def for_owner(self, owner_id: str | uuid.UUID) -> "HydraulicSystemQuerySet":
        return self.filter(owner_id=owner_id)

    def with_prefetch(self) -> "HydraulicSystemQuerySet":
        return self.prefetch_related(
            models.Prefetch(
                "components",
                queryset=SystemComponent.objects.only("id", "system_id", "name"),
            )
        )


class SystemComponentQuerySet(models.QuerySet["SystemComponent"]):
    def with_system(self) -> "SystemComponentQuerySet":
        return self.select_related("system")

    def for_system(self, system_id: str | uuid.UUID) -> "SystemComponentQuerySet":
        return self.filter(system_id=system_id)


class SensorDataQuerySet(models.QuerySet["SensorData"]):
    def for_system(self, system_id: str | uuid.UUID) -> "SensorDataQuerySet":
        return self.filter(system_id=system_id).select_related("component")

    def for_component(self, component_id: str | uuid.UUID) -> "SensorDataQuerySet":
        return self.filter(component_id=component_id)

    def time_range(self, start: datetime, end: datetime) -> "SensorDataQuerySet":
        return self.filter(timestamp__gte=start, timestamp__lt=end)

    def recent(self, hours: int = 24) -> "SensorDataQuerySet":
        return self.filter(timestamp__gte=timezone.now() - timedelta(hours=hours))

    def with_system_component(self) -> "SensorDataQuerySet":
        return self.select_related("system", "component")


class HydraulicSystem(models.Model):
    SYSTEM_TYPES = [
        ("industrial", "Промышленная"),
        ("mobile", "Мобильная"),
        ("marine", "Морская"),
        ("aviation", "Авиационная"),
        ("construction", "Строительная"),
        ("mining", "Горнодобывающая"),
        ("agricultural", "Сельскохозяйственная"),
    ]
    STATUS_CHOICES = [
        ("active", "Активна"),
        ("maintenance", "На обслуживании"),
        ("inactive", "Неактивна"),
        ("emergency", "Аварийная"),
        ("decommissioned", "Списана"),
    ]

    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name: models.CharField = models.CharField(max_length=200, db_index=True)
    description: models.TextField = models.TextField(blank=True, default="")
    system_type: models.CharField = models.CharField(max_length=50, choices=SYSTEM_TYPES, db_index=True)
    status: models.CharField = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active", db_index=True)

    owner: models.ForeignKey = models.ForeignKey("users.User", on_delete=models.PROTECT, related_name="hydraulic_systems", db_index=True)

    criticality: models.CharField = models.CharField(max_length=20, default="medium", db_index=True)
    location: models.CharField = models.CharField(max_length=200, blank=True, default="")
    installation_date: models.DateField = models.DateField(null=True, blank=True)

    components_count: models.PositiveIntegerField = models.PositiveIntegerField(default=0)
    last_reading_at: models.DateTimeField = models.DateTimeField(null=True, blank=True, db_index=True)

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True, db_index=True)

    objects = models.Manager()
    qs: HydraulicSystemQuerySet = HydraulicSystemQuerySet.as_manager()  # type: ignore[assignment]

    if TYPE_CHECKING:
        components: RelatedManager["SystemComponent"]
        sensor_data: RelatedManager["SensorData"]

    class Meta:
        db_table = "diagnostics_hydraulicsystem"
        ordering = ["-updated_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.system_type})"


class SystemComponent(models.Model):
    id: models.UUIDField = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system: models.ForeignKey = models.ForeignKey(HydraulicSystem, related_name="components", on_delete=models.CASCADE, db_index=True)
    name: models.CharField = models.CharField(max_length=255)

    created_at: models.DateTimeField = models.DateTimeField(default=timezone.now, db_index=True)
    updated_at: models.DateTimeField = models.DateTimeField(auto_now=True)

    objects = models.Manager()
    qs: SystemComponentQuerySet = SystemComponentQuerySet.as_manager()  # type: ignore[assignment]

    if TYPE_CHECKING:
        sensor_data: RelatedManager["SensorData"]

    class Meta:
        db_table = "diagnostics_systemcomponent"
        ordering = ["name"]

    def __str__(self) -> str:
        # mypy-safe: str casts
        sys_name = str(getattr(self.system, "name", ""))
        comp_name = str(getattr(self, "name", ""))
        return f"{sys_name}::{comp_name}"
