"""Django signals for diagnostics models (typed)."""
from __future__ import annotations

from typing import Any, Optional, Type

from django.db import models
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import HydraulicSystem, SensorData, SystemComponent


@receiver(post_save, sender=SystemComponent)
def update_components_count_on_create(
    sender: Type[SystemComponent], instance: SystemComponent, created: bool, **kwargs: Any
) -> None:
    """Update components count when component is created."""
    if created and instance.system_id:
        HydraulicSystem.objects.filter(id=instance.system_id).update(
            components_count=models.F("components_count") + 1
        )


@receiver(post_delete, sender=SystemComponent)
def update_components_count_on_delete(
    sender: Type[SystemComponent], instance: SystemComponent, **kwargs: Any
) -> None:
    """Update components count when component is deleted."""
    if instance.system_id:
        HydraulicSystem.objects.filter(id=instance.system_id).update(
            components_count=models.F("components_count") - 1
        )


@receiver(post_save, sender=SensorData)
def update_last_reading_at(
    sender: Type[SensorData], instance: SensorData, created: bool, **kwargs: Any
) -> None:
    """Update system last_reading_at when new sensor data is saved."""
    if created and instance.system_id:
        HydraulicSystem.objects.filter(id=instance.system_id).update(
            last_reading_at=models.Case(
                models.When(
                    models.Q(last_reading_at__lt=instance.timestamp)
                    | models.Q(last_reading_at__isnull=True),
                    then=models.Value(instance.timestamp),
                ),
                default=models.F("last_reading_at"),
            )
        )
