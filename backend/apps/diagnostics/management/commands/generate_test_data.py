import json
import logging
import random
from datetime import datetime, timedelta

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand

from apps.diagnostics.models import DiagnosticReport, HydraulicSystem, SensorData

User = get_user_model()
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Генерация реалистичных тестовых данных для гидравлических систем"

    def add_arguments(self, parser):
        parser.add_argument(
            "--systems", type=int, default=5, help="Количество систем для создания"
        )
        parser.add_argument(
            "--sensors", type=int, default=100, help="Записей датчиков на систему"
        )
        parser.add_argument("--days", type=int, default=7, help="Период данных в днях")
        parser.add_argument(
            "--user-id",
            type=int,
            default=None,
            help="ID пользователя (или первый доступный)",
        )

    def handle(self, *args, **options):  # noqa: C901
        systems_count = options["systems"]
        sensors_count = options["sensors"]
        days_back = options["days"]
        user_id = options["user_id"]
        try:
            # Получение пользователя
            if user_id:
                user = User.objects.get(id=user_id)
            else:
                user = User.objects.first()

            if not user:
                self.stdout.write(
                    self.style.ERROR(
                        "Пользователь не найден. Создайте пользователя сначала."
                    )
                )
                return
            # ... (остальная функция без изменений)
            # Итоговая статистика
            total_reports = DiagnosticReport.objects.filter(system__owner=user).count()
            self.stdout.write(
                self.style.SUCCESS(
                    "\n✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА:\n"
                    f"👥 Пользователь: {user.username}\n"
                    f"🏭 Создано систем: {systems_count}\n"
                    f"📊 Создано записей датчиков: {total_sensors_created}\n"
                    f"📋 Создано отчетов: {total_reports}\n"
                    f"⏱ Период данных: {days_back} дней\n"
                )
            )
        except Exception as e:
            logger.error(f"Ошибка генерации тестовых данных: {e}")
            self.stdout.write(self.style.ERROR(f"❌ Ошибка генерации: {str(e)}"))
