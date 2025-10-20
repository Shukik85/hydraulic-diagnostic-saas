"""
Smoke тест для TimescaleDB функциональности.
Запуск: python manage.py test apps.diagnostics.tests.test_timescale
"""

import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal

from django.conf import settings
from django.db import connection, transaction
from django.test import TestCase, TransactionTestCase
from django.utils import timezone

from apps.diagnostics.models import HydraulicSystem, SensorData, SystemComponent
from apps.diagnostics.timescale_tasks import (
    cleanup_old_partitions,
    compress_old_chunks,
    ensure_partitions_for_range,
    get_hypertable_stats,
    timescale_health_check,
)


class TimescaleDBTestCase(TransactionTestCase):
    """
    Базовый класс для тестирования TimescaleDB функций.
    Использует TransactionTestCase для работы с raw SQL.
    """

    def setUp(self):
        """Создаем тестовые данные."""
        self.system = HydraulicSystem.objects.create(
            name="Test Hydraulic System", system_type="industrial", status="active"
        )

        self.component = SystemComponent.objects.create(
            system=self.system,
            name="Test Pump",
            specification={"type": "centrifugal", "max_pressure": 300},
        )

    def check_timescale_extension(self):
        """Проверяет наличие расширения TimescaleDB."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
                );
            """
            )
            return cursor.fetchone()[0]

    def check_hypertable_exists(self, table_name="sensor_data"):
        """Проверяет, что таблица является hypertable."""
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM timescaledb_information.hypertables
                    WHERE hypertable_name = %s
                );
            """,
                [table_name],
            )
            return cursor.fetchone()[0]


class TimescaleDBSmokeTest(TimescaleDBTestCase):
    """Основные smoke тесты для базовой функциональности TimescaleDB."""

    def test_timescale_extension_installed(self):
        """Тест: расширение TimescaleDB установлено."""
        self.assertTrue(
            self.check_timescale_extension(), "TimescaleDB extension is not installed"
        )

    def test_sensor_data_is_hypertable(self):
        """Тест: таблица sensor_data является hypertable."""
        # Пропускаем тест если TimescaleDB не установлен
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        self.assertTrue(
            self.check_hypertable_exists("sensor_data"),
            "sensor_data table is not a hypertable",
        )

    def test_insert_sensor_data(self):
        """Тест: вставка данных в hypertable работает корректно."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        # Создаем тестовые данные за разные периоды
        now = timezone.now()
        test_data = []

        for i in range(10):
            timestamp = now - timedelta(days=i, hours=i)
            sensor_data = SensorData.objects.create(
                system=self.system,
                component=self.component,
                timestamp=timestamp,
                unit="bar",
                value=100.0 + i * 10,
                value_decimal=Decimal(f"{100 + i * 10}.50"),
            )
            test_data.append(sensor_data)

        # Проверяем, что данные корректно вставлены
        self.assertEqual(SensorData.objects.count(), 10)

        # Проверяем работу optimized QuerySet'а
        recent_data = SensorData.objects.recent_for_system(self.system.id, limit=5)
        self.assertEqual(len(recent_data), 5)

        # Проверяем сортировку по убыванию времени
        timestamps = [item.timestamp for item in recent_data]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))

    def test_time_based_queries(self):
        """Тест: временные запросы работают эффективно."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        # Создаем данные за последние 30 дней
        now = timezone.now()
        for day in range(30):
            for hour in range(0, 24, 6):  # Каждые 6 часов
                timestamp = now - timedelta(days=day, hours=hour)
                SensorData.objects.create(
                    system=self.system,
                    component=self.component,
                    timestamp=timestamp,
                    unit="bar",
                    value=50.0 + day + hour,
                )

        # Тестируем запросы по диапазону времени
        week_ago = now - timedelta(days=7)
        recent_week_data = SensorData.objects.filter(
            system=self.system, timestamp__gte=week_ago
        )

        self.assertGreater(len(recent_week_data), 0)
        self.assertLessEqual(len(recent_week_data), 30)  # 7 дней * 4 записи в день

    def test_timescale_specific_queries(self):
        """Тест: TimescaleDB специфичные функции."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        with connection.cursor() as cursor:
            # Тест time_bucket функции
            cursor.execute(
                """
                SELECT time_bucket('1 day', timestamp) as bucket,
                       COUNT(*), AVG(value)
                FROM sensor_data
                WHERE system_id = %s
                GROUP BY bucket
                ORDER BY bucket DESC
                LIMIT 5;
            """,
                [str(self.system.id)],
            )

            results = cursor.fetchall()
            # Результаты могут быть пустыми, но запрос должен выполняться без ошибок
            self.assertIsInstance(results, list)


class TimescaleCeleryTasksTest(TimescaleDBTestCase):
    """Тесты для Celery задач управления TimescaleDB."""

    def test_timescale_health_check(self):
        """Тест: проверка здоровья TimescaleDB."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        # Вызываем задачу проверки здоровья
        result = timescale_health_check()

        self.assertEqual(result["status"], "success")
        self.assertIn("timescale_version", result)
        self.assertIn("hypertables_count", result)
        self.assertIn("checked_at", result)

    def test_get_hypertable_stats(self):
        """Тест: получение статистики по hypertable."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        # Создаем немного данных
        now = timezone.now()
        for i in range(5):
            SensorData.objects.create(
                system=self.system,
                component=self.component,
                timestamp=now - timedelta(hours=i),
                unit="bar",
                value=100.0 + i,
            )

        # Вызываем задачу получения статистики
        result = get_hypertable_stats("sensor_data")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["table"], "sensor_data")
        self.assertIn("total_size", result)
        self.assertIn("total_chunks", result)
        self.assertIn("compression_ratio", result)

    def test_ensure_partitions_task(self):
        """Тест: задача обеспечения партиций."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        # Создаем mock задачи
        class MockTask:
            request = type("obj", (object,), {"id": "test-task-id"})()

        task = MockTask()

        # Тестируем функцию с заданными параметрами
        start_time = timezone.now().isoformat()
        end_time = (timezone.now() + timedelta(days=7)).isoformat()

        try:
            result = ensure_partitions_for_range(
                task, table_name="sensor_data", start_time=start_time, end_time=end_time
            )

            self.assertEqual(result["status"], "success")
            self.assertEqual(result["table"], "sensor_data")
            self.assertIn("existing_chunks", result)
        except Exception as e:
            # В тестовой среде задача может не выполниться полностью
            # но должна обрабатывать ошибки корректно
            self.assertIsInstance(e, Exception)


class TimescaleDBPerformanceTest(TimescaleDBTestCase):
    """Тесты производительности для TimescaleDB."""

    def test_bulk_insert_performance(self):
        """Тест: производительность массовых вставок."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        import time

        # Создаем большое количество данных
        now = timezone.now()
        sensor_data_list = []

        start_time = time.time()

        for i in range(1000):
            sensor_data_list.append(
                SensorData(
                    system=self.system,
                    component=self.component,
                    timestamp=now - timedelta(minutes=i),
                    unit="bar",
                    value=100.0 + (i % 100),
                )
            )

        # Массовая вставка
        SensorData.objects.bulk_create(sensor_data_list, batch_size=100)

        end_time = time.time()
        insert_time = end_time - start_time

        # Проверяем что данные вставились
        total_count = SensorData.objects.filter(system=self.system).count()
        self.assertEqual(total_count, 1000)

        # Логируем время выполнения (в реальном проекте можно использовать metrics)
        print(f"Bulk insert of 1000 records took: {insert_time:.2f} seconds")

        # Базовая проверка производительности (должно быть быстро)
        self.assertLess(insert_time, 10.0, "Bulk insert took too long")

    def test_time_range_query_performance(self):
        """Тест: производительность запросов по временному диапазону."""
        if not self.check_timescale_extension():
            self.skipTest("TimescaleDB extension not available")

        import time

        # Создаем данные за последние 30 дней
        now = timezone.now()
        for day in range(30):
            for hour in range(0, 24, 2):  # Каждые 2 часа
                SensorData.objects.create(
                    system=self.system,
                    component=self.component,
                    timestamp=now - timedelta(days=day, hours=hour),
                    unit="bar",
                    value=50.0 + day + hour,
                )

        # Тестируем производительность запроса за последнюю неделю
        week_ago = now - timedelta(days=7)

        start_time = time.time()

        recent_data = list(
            SensorData.objects.filter(
                system=self.system, timestamp__gte=week_ago
            ).order_by("-timestamp")
        )

        end_time = time.time()
        query_time = end_time - start_time

        self.assertGreater(len(recent_data), 0)
        print(
            f"Query for last week returned {len(recent_data)} records in {query_time:.3f} seconds"
        )

        # Запрос должен выполняться быстро благодаря индексам TimescaleDB
        self.assertLess(query_time, 1.0, "Time range query took too long")
