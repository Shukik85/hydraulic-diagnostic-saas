"""Django signals for diagnostics models (typed, mypy-safe)."""

from __future__ import annotations

from typing import Any

from django.db import models
from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from .models import HydraulicSystem, SensorData, SystemComponent


@receiver(post_save, sender=SystemComponent)
def update_components_count_on_create(
    sender: type[SystemComponent],
    instance: SystemComponent,
    created: bool,
    **kwargs: Any,
) -> None:
    """Выполняет update components count on create
    Args:
        sender (Any): Параметр sender
        instance (Any): Параметр instance
        created (Any): Параметр created.

    """
    if created:
        sys_pk = getattr(instance.system, "pk", None)
        if sys_pk is not None:
            HydraulicSystem.objects.filter(id=sys_pk).update(
                components_count=models.F("components_count") + 1
            )


@receiver(post_delete, sender=SystemComponent)
def update_components_count_on_delete(
    sender: type[SystemComponent], instance: SystemComponent, **kwargs: Any
) -> None:
    """Выполняет update components count on delete.

    pass
    Args:
        sender (Any): Параметр sender
        instance (Any): Параметр instance

    """
    sys_pk = getattr(instance.system, "pk", None)
    if sys_pk is not None:
        HydraulicSystem.objects.filter(id=sys_pk).update(
            components_count=models.F("components_count") - 1
        )


@receiver(post_save, sender=SensorData)
def update_last_reading_at(
    sender: type[SensorData], instance: SensorData, created: bool, **kwargs: Any
) -> None:
    """Выполняет update last reading at.

    pass
    Args:
        sender (Any): Параметр sender
        instance (Any): Параметр instance
        created (Any): Параметр created

    """
    if created:
        sys_pk = getattr(instance.system, "pk", None)
        if sys_pk is not None:
            HydraulicSystem.objects.filter(id=sys_pk).update(
                last_reading_at=models.Case(
                    models.When(
                        models.Q(last_reading_at__lt=instance.timestamp)
                        | models.Q(last_reading_at__isnull=True),
                        then=models.Value(instance.timestamp),
                    ),
                    default=models.F("last_reading_at"),
                )
            )
