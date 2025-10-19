from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

from .models import SystemComponent, SensorData, HydraulicSystem


@receiver(post_save, sender=SystemComponent)
def update_components_count_on_create(sender, instance: SystemComponent, created, **kwargs):
    if created:
        HydraulicSystem.objects.filter(id=instance.system_id).update(
            components_count=models.F("components_count") + 1
        )


@receiver(post_delete, sender=SystemComponent)
def update_components_count_on_delete(sender, instance: SystemComponent, **kwargs):
    HydraulicSystem.objects.filter(id=instance.system_id).update(
        components_count=models.F("components_count") - 1
    )


@receiver(post_save, sender=SensorData)
def update_last_reading_at(sender, instance: SensorData, created, **kwargs):
    if created:
        # Обновляем last_reading_at максимальным значением
        HydraulicSystem.objects.filter(id=instance.system_id).update(
            last_reading_at=models.Case(
                models.When(
                    models.Q(last_reading_at__lt=instance.timestamp) | models.Q(last_reading_at__isnull=True),
                    then=models.Value(instance.timestamp),
                ),
                default=models.F("last_reading_at"),
            )
        )
