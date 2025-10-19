import logging
from datetime import datetime, timedelta

from django.db.models.signals import post_save, pre_delete
from django.dispatch import receiver

from .models import DiagnosticReport, HydraulicSystem, SensorData

logger = logging.getLogger(__name__)


@receiver(post_save, sender=SensorData)
def process_critical_sensor_data(sender, instance, created, **kwargs):
    """Обработка критических данных датчиков"""
    if created and instance.is_critical:
        try:
            # Логирование критического события
            logger.warning(
                f"🚨 Критическое событие: {instance.system.name} - "
                f"{instance.sensor_type}: {instance.value} {instance.unit}"
            )

            # Проверка на множественные критические события
            recent_critical = SensorData.objects.filter(
                system=instance.system,
                is_critical=True,
                timestamp__gte=datetime.now() - timedelta(hours=1),
            ).count()

            # Автоматическое создание отчета при множественных проблемах
            if recent_critical >= 5:  # 5 критических событий за час
                # Проверяем, есть ли уже отчет за последний час
                recent_reports = DiagnosticReport.objects.filter(
                    system=instance.system,
                    created_at__gte=datetime.now() - timedelta(hours=1),
                    title__icontains="Автоматический отчет",
                ).count()

                if recent_reports == 0:
                    DiagnosticReport.objects.create(
                        system=instance.system,
                        title=(
                            "Автоматический отчет - "
                            "Множественные критические события"
                        ),
                        description=(
                            f"Система {instance.system.name} зафиксировала "(
                                f"{recent_critical} критических событий"
                                " за последний час. "
                                "Рекомендуется немедленная проверка."
                            )
                        ),
                        severity="critical",
                    )

                    logger.error(
                        f"🚨 Создан автоматический критический отчет для "
                        f"{instance.system.name}"
                    )

        except Exception as e:
            logger.error(f"Ошибка обработки критических данных: {e}")


@receiver(post_save, sender=HydraulicSystem)
def initialize_system(sender, instance, created, **kwargs):
    """Инициализация новой системы"""
    if created:
        try:
            logger.info(
                f"✅ Создана новая система: {instance.name} "
                f"(владелец: {instance.owner.username})"
            )

            # Можно добавить инициализационные действия:
            # - Создание базовых настроек
            # - Отправка уведомлений
            # - Настройка мониторинга

        except Exception as e:
            logger.error(f"Ошибка инициализации системы {instance.name}: {e}")


@receiver(pre_delete, sender=HydraulicSystem)
def cleanup_system_data(sender, instance, **kwargs):
    """Очистка данных при удалении системы"""
    try:
        # Подсчет удаляемых данных
        sensor_count = instance.sensor_data.count()
        reports_count = instance.diagnostic_reports.count()

        logger.info(
            f"🗑 Удаление системы {instance.name}: "
            f"{sensor_count} записей датчиков, {reports_count} отчетов"
        )

    except Exception as e:
        logger.error(f"Ошибка при удалении системы {instance.name}: {e}")


@receiver(post_save, sender=DiagnosticReport)
def process_diagnostic_report(sender, instance, created, **kwargs):
    """Обработка диагностических отчетов"""
    if created:
        try:
            logger.info(
                f"📋 Создан отчет: {instance.title} "(
                    f"(система: {instance.system.name}, "
                    f"серьезность: {instance.severity})"
                )
            )

            # Дополнительные действия для критических отчетов
            if instance.severity == "critical":
                # Здесь можно добавить:
                # - Отправку уведомлений
                # - Создание заявок на обслуживание
                # - Автоматическое изменение статуса системы
                pass

        except Exception as e:
            logger.error(f"Ошибка обработки отчета {instance.title}: {e}")
