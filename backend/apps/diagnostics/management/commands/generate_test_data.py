"""Модуль проекта с автогенерированным докстрингом."""

from datetime import datetime, timedelta
import json
import logging
import random

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

    def _get_user(self, user_id):
        """Получение пользователя по ID или первого доступного."""
        if user_id:
            return User.objects.get(id=user_id)
        return User.objects.first()

    def _get_system_templates(self):
        """Шаблоны систем с реалистичными параметрами."""
        return [
            {
                "name_pattern": "Промышленный пресс №{}",
                "type": "industrial",
                "location_pattern": "Цех {} - Линия производства",
                "pressure_range": (180.0, 320.0),
                "flow_range": (45.0, 85.0),
                "temp_range": (45.0, 65.0),
                "vibration_range": (2.0, 12.0),
            },
            {
                "name_pattern": "Мобильная система №{}",
                "type": "mobile",
                "location_pattern": "Строительная площадка {} - Экскаватор",
                "pressure_range": (220.0, 380.0),
                "flow_range": (60.0, 120.0),
                "temp_range": (40.0, 75.0),
                "vibration_range": (5.0, 18.0),
            },
            {
                "name_pattern": "Морская система №{}",
                "type": "marine",
                "location_pattern": "Порт {} - Кран перегрузочный",
                "pressure_range": (200.0, 300.0),
                "flow_range": (35.0, 75.0),
                "temp_range": (35.0, 60.0),
                "vibration_range": (3.0, 15.0),
            },
            {
                "name_pattern": "Авиационная система №{}",
                "type": "aviation",
                "location_pattern": "Аэропорт {} - Гидросистема шасси",
                "pressure_range": (280.0, 420.0),
                "flow_range": (25.0, 55.0),
                "temp_range": (30.0, 80.0),
                "vibration_range": (1.0, 8.0),
            },
        ]

    def _create_hydraulic_system(self, template, system_num, user):
        """Создание одной гидравлической системы."""
        statuses = ["active", "maintenance", "inactive"]
        status_weights = [0.7, 0.2, 0.1]

        status = random.choices(statuses, weights=status_weights)[0]
        location_num = random.randint(1, 50)

        return HydraulicSystem.objects.create(
            name=template["name_pattern"].format(system_num),
            system_type=template["type"],
            location=template["location_pattern"].format(location_num),
            status=status,
            max_pressure=random.uniform(*template["pressure_range"]),
            flow_rate=random.uniform(*template["flow_range"]),
            temperature_range="-20°C до +85°C",
            owner=user,
            installation_date=datetime.now().date()
            - timedelta(days=random.randint(30, 1000)),
        )

    def _calculate_sensor_values(self, base_values, daily_offsets, drifts, noises):
        """Расчет значений датчиков с учетом всех факторов."""
        current_pressure = (
            base_values["pressure"]
            + daily_offsets["pressure"]
            + drifts["pressure"]
            + noises["pressure"]
        )
        current_temp = (
            base_values["temp"]
            + daily_offsets["temp"]
            + drifts["temp"]
            + noises["temp"]
        )
        current_flow = base_values["flow"] + daily_offsets["flow"] + noises["flow"]
        current_vibration = (
            base_values["vibration"] + drifts["vibration"] + noises["vibration"]
        )

        # Ограничение значений в реалистичных пределах
        return {
            "pressure": max(50, min(500, current_pressure)),
            "temp": max(10, min(100, current_temp)),
            "flow": max(5, min(200, current_flow)),
            "vibration": max(0, min(50, current_vibration)),
        }

    def _determine_critical_conditions(self, values, system, base_flow):
        """Определение критических условий для датчиков."""
        return {
            "pressure": values["pressure"] > system.max_pressure * 0.95
            or values["pressure"] < 80,
            "temp": values["temp"] > 85 or values["temp"] < 15,
            "flow": values["flow"] < base_flow * 0.7,
            "vibration": values["vibration"] > 25,
        }

    def _create_sensor_data_objects(
        self, system, timestamp, values, critical_conditions
    ):
        """Создание объектов данных датчиков."""
        sensor_types_data = [
            {
                "sensor_type": "pressure",
                "value": round(values["pressure"], 2),
                "unit": "bar",
                "is_critical": critical_conditions["pressure"],
                "warning_message": (
                    "Критическое давление!" if critical_conditions["pressure"] else ""
                ),
            },
            {
                "sensor_type": "temperature",
                "value": round(values["temp"], 1),
                "unit": "°C",
                "is_critical": critical_conditions["temp"],
                "warning_message": (
                    "Критическая температура!" if critical_conditions["temp"] else ""
                ),
            },
            {
                "sensor_type": "flow",
                "value": round(values["flow"], 2),
                "unit": "л/мин",
                "is_critical": critical_conditions["flow"],
                "warning_message": (
                    "Низкий расход!" if critical_conditions["flow"] else ""
                ),
            },
            {
                "sensor_type": "vibration",
                "value": round(values["vibration"], 2),
                "unit": "мм/с",
                "is_critical": critical_conditions["vibration"],
                "warning_message": (
                    "Высокая вибрация!" if critical_conditions["vibration"] else ""
                ),
            },
        ]

        return [
            SensorData(system=system, timestamp=timestamp, **sensor_data)
            for sensor_data in sensor_types_data
        ]

    def _generate_sensor_data_for_system(
        self, system, template, sensors_count, days_back
    ):
        """Генерация данных датчиков для одной системы."""
        sensors_for_system = []
        start_time = datetime.now() - timedelta(days=days_back)

        # Базовые значения для системы
        base_values = {
            "pressure": random.uniform(*template["pressure_range"]),
            "flow": random.uniform(*template["flow_range"]),
            "temp": random.uniform(*template["temp_range"]),
            "vibration": random.uniform(*template["vibration_range"]),
        }

        intervals_per_day = max(1, sensors_count // days_back)

        # Факторы деградации
        drifts_per_day = {
            "pressure": random.uniform(-0.5, 0.5),
            "temp": random.uniform(0, 0.2),
            "vibration": random.uniform(0, 0.1),
        }

        for day in range(days_back):
            day_start = start_time + timedelta(days=day)

            # Дневные колебания
            daily_offsets = {
                "pressure": random.uniform(-20, 20),
                "temp": random.uniform(-5, 15),
                "flow": random.uniform(-10, 10),
            }

            # Влияние деградации
            degradation_days = day
            current_drifts = {
                key: value * degradation_days for key, value in drifts_per_day.items()
            }

            for _ in range(intervals_per_day):
                timestamp = day_start + timedelta(
                    hours=random.uniform(6, 22),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59),
                )

                # Случайные колебания
                noises = {
                    "pressure": random.uniform(-15, 15),
                    "temp": random.uniform(-3, 8),
                    "flow": random.uniform(-8, 8),
                    "vibration": random.uniform(-2, 5),
                }

                values = self._calculate_sensor_values(
                    base_values, daily_offsets, current_drifts, noises
                )
                critical_conditions = self._determine_critical_conditions(
                    values, system, base_values["flow"]
                )

                sensors_for_system.extend(
                    self._create_sensor_data_objects(
                        system, timestamp, values, critical_conditions
                    )
                )

        return sensors_for_system

    def _create_diagnostic_report(self, system, day, events):
        """Создание диагностического отчета для дня с критическими событиями."""
        severity = (
            "critical"
            if len(events) >= 10
            else "error" if len(events) >= 6 else "warning"
        )

        # Анализ типов проблем
        problem_types = {}
        for event in events:
            problem_types[event.sensor_type] = (
                problem_types.get(event.sensor_type, 0) + 1
            )

        # Формирование описания
        sensor_names = {
            "pressure": "давление",
            "temperature": "температура",
            "flow": "расход",
            "vibration": "вибрация",
        }
        problems_desc = [
            f"{sensor_names.get(sensor_type, sensor_type)}: {count} событий"
            for sensor_type, count in problem_types.items()
        ]

        DiagnosticReport.objects.create(
            system=system,
            title=f"Критические события {day.strftime('%d.%m.%Y')}",
            description=(
                f"Обнаружено {len(events)} критических событий: {', '.join(problems_desc)}. "
                "Рекомендуется проверка системы и возможное техническое обслуживание."
            ),
            severity=severity,
            ai_analysis=json.dumps(
                {
                    "analysis_date": day.isoformat(),
                    "critical_events_count": len(events),
                    "problem_breakdown": problem_types,
                    "recommendations": [
                        "Проверить состояние фильтров",
                        "Измерить температуру масла",
                        "Проверить герметичность соединений",
                        "Проанализировать нагрузку на систему",
                    ],
                    "urgency_level": severity,
                    "estimated_repair_time": (
                        "2-4 часа" if severity == "warning" else "4-8 часов"
                    ),
                },
                ensure_ascii=False,
            ),
        )

    def _generate_diagnostic_reports(self, system):
        """Генерация диагностических отчетов на основе критических событий."""
        critical_events = SensorData.objects.filter(
            system=system, is_critical=True
        ).order_by("timestamp")

        if not critical_events.exists():
            return

        # Группировка критических событий по дням
        events_by_day = {}
        for event in critical_events:
            day_key = event.timestamp.date()
            if day_key not in events_by_day:
                events_by_day[day_key] = []
            events_by_day[day_key].append(event)

        # Создание отчетов для дней с множественными проблемами
        for day, events in events_by_day.items():
            if len(events) >= 3:
                self._create_diagnostic_report(system, day, events)

    def _display_summary(self, user, systems_count, total_sensors_created, days_back):
        """Отображение итоговой статистики."""
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

    def handle(self, *args, **options):
        systems_count = options["systems"]
        sensors_count = options["sensors"]
        days_back = options["days"]
        user_id = options["user_id"]
        try:
            user = self._get_user(user_id)
            if not user:
                self.stdout.write(
                    self.style.ERROR(
                        "Пользователь не найден. Создайте пользователя сначала."
                    )
                )
                return

            system_templates = self._get_system_templates()
            created_systems = []

            # Создание систем
            for i in range(systems_count):
                template = random.choice(system_templates)
                system = self._create_hydraulic_system(template, i + 1, user)
                created_systems.append((system, template))
                self.stdout.write(f"Создана система: {system.name}")

            # Генерация данных датчиков
            total_sensors_created = 0
            for system, template in created_systems:
                self.stdout.write(f"Генерация данных для {system.name}...")

                sensors_for_system = self._generate_sensor_data_for_system(
                    system, template, sensors_count, days_back
                )

                if sensors_for_system:
                    SensorData.objects.bulk_create(sensors_for_system, batch_size=500)
                    total_sensors_created += len(sensors_for_system)
                    self.stdout.write(
                        f"  → Создано {len(sensors_for_system)} записей датчиков"
                    )

                # Генерация диагностических отчетов
                self._generate_diagnostic_reports(system)

            # Отображение итогов
            self._display_summary(user, systems_count, total_sensors_created, days_back)

        except Exception as e:
            logger.error(f"Ошибка генерации тестовых данных: {e}")
            self.stdout.write(self.style.ERROR(f"❌ Ошибка генерации: {e!s}"))
