import random
import json
from datetime import datetime, timedelta
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from apps.diagnostics.models import HydraulicSystem, SensorData, DiagnosticReport
import logging

User = get_user_model()
logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = 'Генерация реалистичных тестовых данных для гидравлических систем'

    def add_arguments(self, parser):
        parser.add_argument('--systems', type=int, default=5,
                            help='Количество систем для создания')
        parser.add_argument('--sensors', type=int, default=100,
                            help='Записей датчиков на систему')
        parser.add_argument('--days', type=int, default=7,
                            help='Период данных в днях')
        parser.add_argument('--user-id', type=int, default=None,
                            help='ID пользователя (или первый доступный)')

    def handle(self, *args, **options):
        systems_count = options['systems']
        sensors_count = options['sensors']
        days_back = options['days']
        user_id = options['user_id']

        try:
            # Получение пользователя
            if user_id:
                user = User.objects.get(id=user_id)
            else:
                user = User.objects.first()

            if not user:
                self.stdout.write(self.style.ERROR(
                    'Пользователь не найден. Создайте пользователя сначала.'))
                return

            # Шаблоны систем с реалистичными параметрами
            system_templates = [
                {
                    'name_pattern': 'Промышленный пресс №{}',
                    'type': 'industrial',
                    'location_pattern': 'Цех {} - Линия производства',
                    'pressure_range': (180.0, 320.0),
                    'flow_range': (45.0, 85.0),
                    'temp_range': (45.0, 65.0),
                    'vibration_range': (2.0, 12.0)
                },
                {
                    'name_pattern': 'Мобильная система №{}',
                    'type': 'mobile',
                    'location_pattern': 'Строительная площадка {} - Экскаватор',
                    'pressure_range': (220.0, 380.0),
                    'flow_range': (60.0, 120.0),
                    'temp_range': (40.0, 75.0),
                    'vibration_range': (5.0, 18.0)
                },
                {
                    'name_pattern': 'Морская система №{}',
                    'type': 'marine',
                    'location_pattern': 'Порт {} - Кран перегрузочный',
                    'pressure_range': (200.0, 300.0),
                    'flow_range': (35.0, 75.0),
                    'temp_range': (35.0, 60.0),
                    'vibration_range': (3.0, 15.0)
                },
                {
                    'name_pattern': 'Авиационная система №{}',
                    'type': 'aviation',
                    'location_pattern': 'Аэропорт {} - Гидросистема шасси',
                    'pressure_range': (280.0, 420.0),
                    'flow_range': (25.0, 55.0),
                    'temp_range': (30.0, 80.0),
                    'vibration_range': (1.0, 8.0)
                }
            ]

            statuses = ['active', 'maintenance', 'inactive']
            status_weights = [0.7, 0.2, 0.1]  # Большинство активных систем

            created_systems = []

            # Создание систем
            for i in range(systems_count):
                template = random.choice(system_templates)
                status = random.choices(statuses, weights=status_weights)

                # Генерация уникальных данных
                location_num = random.randint(1, 50)
                system_num = i + 1

                system = HydraulicSystem.objects.create(
                    name=template['name_pattern'].format(system_num),
                    system_type=template['type'],
                    location=template['location_pattern'].format(location_num),
                    status=status,
                    max_pressure=random.uniform(*template['pressure_range']),
                    flow_rate=random.uniform(*template['flow_range']),
                    temperature_range='-20°C до +85°C',
                    owner=user,
                    installation_date=datetime.now().date() - timedelta(days=random.randint(30, 1000))
                )
                created_systems.append((system, template))

                self.stdout.write(f'Создана система: {system.name}')

            # Генерация реалистичных данных датчиков
            total_sensors_created = 0

            for system, template in created_systems:
                self.stdout.write(f'Генерация данных для {system.name}...')

                sensors_for_system = []
                start_time = datetime.now() - timedelta(days=days_back)

                # Базовые значения для системы (вокруг них будут колебания)
                base_pressure = random.uniform(*template['pressure_range'])
                base_flow = random.uniform(*template['flow_range'])
                base_temp = random.uniform(*template['temp_range'])
                base_vibration = random.uniform(*template['vibration_range'])

                # Симуляция работы системы по дням
                current_time = start_time
                intervals_per_day = max(1, sensors_count // days_back)

                # Факторы деградации (имитация износа)
                pressure_degradation = random.uniform(-0.5, 0.5)  # бар/день
                temp_increase = random.uniform(0, 0.2)  # °C/день
                vibration_increase = random.uniform(0, 0.1)  # мм/с/день

                for day in range(days_back):
                    day_start = start_time + timedelta(days=day)

                    # Дневные колебания (имитация рабочих циклов)
                    daily_pressure_offset = random.uniform(-20, 20)
                    daily_temp_offset = random.uniform(-5, 15)
                    daily_flow_offset = random.uniform(-10, 10)

                    # Влияние деградации
                    degradation_days = day
                    pressure_drift = pressure_degradation * degradation_days
                    temp_drift = temp_increase * degradation_days
                    vibration_drift = vibration_increase * degradation_days

                    # Генерация интервалов в течение дня
                    for interval in range(intervals_per_day):
                        timestamp = day_start + timedelta(
                            hours=random.uniform(6, 22),  # Рабочие часы
                            minutes=random.randint(0, 59),
                            seconds=random.randint(0, 59)
                        )

                        # Случайные колебания в пределах нормы
                        pressure_noise = random.uniform(-15, 15)
                        temp_noise = random.uniform(-3, 8)
                        flow_noise = random.uniform(-8, 8)
                        vibration_noise = random.uniform(-2, 5)

                        # Рассчет итоговых значений
                        current_pressure = base_pressure + \
                            daily_pressure_offset + pressure_drift + pressure_noise
                        current_temp = base_temp + daily_temp_offset + temp_drift + temp_noise
                        current_flow = base_flow + daily_flow_offset + flow_noise
                        current_vibration = base_vibration + vibration_drift + vibration_noise

                        # Ограничение значений в реалистичных пределах
                        current_pressure = max(50, min(500, current_pressure))
                        current_temp = max(10, min(100, current_temp))
                        current_flow = max(5, min(200, current_flow))
                        current_vibration = max(0, min(50, current_vibration))

                        # Определение критичности
                        is_critical_pressure = current_pressure > system.max_pressure * \
                            0.95 or current_pressure < 80
                        is_critical_temp = current_temp > 85 or current_temp < 15
                        is_critical_flow = current_flow < base_flow * 0.7
                        is_critical_vibration = current_vibration > 25

                        # Создание записей датчиков
                        sensor_types_data = [
                            {
                                'sensor_type': 'pressure',
                                'value': round(current_pressure, 2),
                                'unit': 'bar',
                                'is_critical': is_critical_pressure,
                                'warning_message': 'Критическое давление!' if is_critical_pressure else ''
                            },
                            {
                                'sensor_type': 'temperature',
                                'value': round(current_temp, 1),
                                'unit': '°C',
                                'is_critical': is_critical_temp,
                                'warning_message': 'Критическая температура!' if is_critical_temp else ''
                            },
                            {
                                'sensor_type': 'flow',
                                'value': round(current_flow, 2),
                                'unit': 'л/мин',
                                'is_critical': is_critical_flow,
                                'warning_message': 'Низкий расход!' if is_critical_flow else ''
                            },
                            {
                                'sensor_type': 'vibration',
                                'value': round(current_vibration, 2),
                                'unit': 'мм/с',
                                'is_critical': is_critical_vibration,
                                'warning_message': 'Высокая вибрация!' if is_critical_vibration else ''
                            }
                        ]

                        for sensor_data in sensor_types_data:
                            sensors_for_system.append(SensorData(
                                system=system,
                                timestamp=timestamp,
                                **sensor_data
                            ))

                # Массовое создание данных датчиков
                if sensors_for_system:
                    SensorData.objects.bulk_create(
                        sensors_for_system, batch_size=500)
                    total_sensors_created += len(sensors_for_system)
                    self.stdout.write(
                        f'  → Создано {len(sensors_for_system)} записей датчиков')

                # Создание диагностических отчетов на основе критических событий
                critical_events = SensorData.objects.filter(
                    system=system,
                    is_critical=True
                ).order_by('timestamp')

                if critical_events.exists():
                    # Группировка критических событий по дням
                    events_by_day = {}
                    for event in critical_events:
                        day_key = event.timestamp.date()
                        if day_key not in events_by_day:
                            events_by_day[day_key] = []
                        events_by_day[day_key].append(event)

                    # Создание отчетов для дней с множественными проблемами
                    for day, events in events_by_day.items():
                        if len(events) >= 3:  # Минимум 3 критических события
                            severity = 'critical' if len(events) >= 10 else 'error' if len(
                                events) >= 6 else 'warning'

                            # Анализ типов проблем
                            problem_types = {}
                            for event in events:
                                if event.sensor_type not in problem_types:
                                    problem_types[event.sensor_type] = 0
                                problem_types[event.sensor_type] += 1

                            # Формирование описания
                            problems_desc = []
                            for sensor_type, count in problem_types.items():
                                sensor_names = {
                                    'pressure': 'давление',
                                    'temperature': 'температура',
                                    'flow': 'расход',
                                    'vibration': 'вибрация'
                                }
                                problems_desc.append(
                                    f"{sensor_names.get(sensor_type, sensor_type)}: {count} событий")

                            report = DiagnosticReport.objects.create(
                                system=system,
                                title=f"Критические события {day.strftime('%d.%m.%Y')}",
                                description=f"Обнаружено {len(events)} критических событий: {', '.join(problems_desc)}. "
                                f"Рекомендуется проверка системы и возможное техническое обслуживание.",
                                severity=severity,
                                ai_analysis=json.dumps({
                                    'analysis_date': day.isoformat(),
                                    'critical_events_count': len(events),
                                    'problem_breakdown': problem_types,
                                    'recommendations': [
                                        'Проверить состояние фильтров',
                                        'Измерить температуру масла',
                                        'Проверить герметичность соединений',
                                        'Проанализировать нагрузку на систему'
                                    ],
                                    'urgency_level': severity,
                                    'estimated_repair_time': '2-4 часа' if severity == 'warning' else '4-8 часов'
                                }, ensure_ascii=False)
                            )

            # Итоговая статистика
            total_reports = DiagnosticReport.objects.filter(
                system__owner=user).count()

            self.stdout.write(
                self.style.SUCCESS(
                    f'\n✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА:\n'
                    f'👥 Пользователь: {user.username}\n'
                    f'🏭 Создано систем: {systems_count}\n'
                    f'📊 Создано записей датчиков: {total_sensors_created}\n'
                    f'📋 Создано отчетов: {total_reports}\n'
                    f'⏱ Период данных: {days_back} дней\n'
                )
            )

        except Exception as e:
            logger.error(f"Ошибка генерации тестовых данных: {e}")
            self.stdout.write(
                self.style.ERROR(f'❌ Ошибка генерации: {str(e)}')
            )
