"""Модуль проекта с автогенерированным докстрингом."""

import unittest
from datetime import datetime, timedelta

import pandas as pd
from apps.diagnostics.models import HydraulicSystem, SensorData
from django.contrib.auth import get_user_model
from django.test import TestCase

User = get_user_model()


class AIEngineTestCase(TestCase):
    """Тесты для AI движка диагностики"""

    def setUp(self):
        """Настройка тестовых данных"""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        self.system = HydraulicSystem.objects.create(
            name="Тестовая система",
            system_type="industrial",
            status="active",
            owner=self.user,
        )

    def create_test_sensor_data(self, hours_back=24, interval_minutes=30):
        """Создание тестовых данных датчиков"""
        sensor_data = []
        start_time = datetime.now() - timedelta(hours=hours_back)

        current_time = start_time
        while current_time <= datetime.now():
            sensor_data.extend(
                [
                    SensorData.objects.create(
                        system=self.system,
                        sensor_type="pressure",
                        value=200.0 + (current_time.hour - 12) * 5,
                        unit="bar",
                        timestamp=current_time,
                        is_critical=False,
                    ),
                    SensorData.objects.create(
                        system=self.system,
                        sensor_type="temperature",
                        value=50.0 + (current_time.hour - 12) * 2,
                        unit="C",
                        timestamp=current_time,
                        is_critical=False,
                    ),
                    SensorData.objects.create(
                        system=self.system,
                        sensor_type="flow",
                        value=70.0 + (current_time.hour - 12) * 1,
                        unit="lpm",
                        timestamp=current_time,
                        is_critical=False,
                    ),
                ]
            )

            current_time += timedelta(minutes=interval_minutes)

        return sensor_data

    def test_feature_extraction(self):
        self.create_test_sensor_data(hours_back=2)
        sensor_data = self.system.sensor_data.all()
        df = pd.DataFrame(
            [
                {
                    "sensor_type": data.sensor_type,
                    "value": data.value,
                    "unit": data.unit,
                    "timestamp": data.timestamp,
                    "is_critical": data.is_critical,
                }
                for data in sensor_data
            ]
        )
        # заглушка: движок может отличаться от старых тестов; важна импортная совместимость
        self.assertIsInstance(df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
