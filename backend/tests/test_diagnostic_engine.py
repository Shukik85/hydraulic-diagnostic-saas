from datetime import datetime
from decimal import Decimal

from django.test import TestCase

from apps.diagnostics.engine import DiagnosticEngine
from apps.diagnostics.models import Equipment, Sensor, SensorData


class DiagnosticEngineTestCase(TestCase):
    """Тесты для DiagnosticEngine"""

    def setUp(self):
        """Настройка тестовых данных"""
        self.equipment = Equipment.objects.create(
            name="Гидравлическая система 1",
            type="hydraulic_system",
            model="HS-100",
            manufacturer="TestManufacturer",
        )

        self.pressure_sensor = Sensor.objects.create(
            equipment=self.equipment,
            name="Датчик давления",
            type="pressure",
            unit="bar",
            min_value=Decimal("0.0"),
            max_value=Decimal("300.0"),
            normal_min=Decimal("50.0"),
            normal_max=Decimal("250.0"),
        )

        self.engine = DiagnosticEngine()

    def test_pressure_anomaly_detection(self):
        """Тест обнаружения аномалии давления"""
        # Создаем данные с аномально высоким давлением
        sensor_data = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("280.0"),  # Выше нормы (250.0)
            timestamp=datetime.now(),
        )

        result = self.engine.analyze(sensor_data)

        self.assertIsNotNone(result)
        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["anomaly_type"], "high_pressure")
        self.assertIn("severity", result)
        self.assertGreater(result["severity"], 0)

    def test_no_anomaly_detection(self):
        """Тест отсутствия аномалий при нормальных значениях"""
        # Создаем данные с нормальным давлением
        sensor_data = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("150.0"),  # В пределах нормы (50.0-250.0)
            timestamp=datetime.now(),
        )

        result = self.engine.analyze(sensor_data)

        self.assertIsNotNone(result)
        self.assertFalse(result["is_anomaly"])
        self.assertEqual(result["anomaly_type"], None)
        self.assertEqual(result["severity"], 0)

    def test_boundary_values_upper(self):
        """Тест пограничных значений - верхняя граница"""
        # Проверяем значение точно на верхней границе нормы
        sensor_data = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("250.0"),  # Точно на границе нормы
            timestamp=datetime.now(),
        )

        result = self.engine.analyze(sensor_data)

        self.assertIsNotNone(result)
        self.assertFalse(result["is_anomaly"])  # На границе еще норма

        # Проверяем значение чуть выше верхней границы
        sensor_data_above = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("250.1"),  # Чуть выше границы нормы
            timestamp=datetime.now(),
        )

        result_above = self.engine.analyze(sensor_data_above)

        self.assertIsNotNone(result_above)
        self.assertTrue(result_above["is_anomaly"])  # Выше границы - аномалия

    def test_boundary_values_lower(self):
        """Тест пограничных значений - нижняя граница"""
        # Проверяем значение точно на нижней границе нормы
        sensor_data = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("50.0"),  # Точно на границе нормы
            timestamp=datetime.now(),
        )

        result = self.engine.analyze(sensor_data)

        self.assertIsNotNone(result)
        self.assertFalse(result["is_anomaly"])  # На границе еще норма

        # Проверяем значение чуть ниже нижней границы
        sensor_data_below = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("49.9"),  # Чуть ниже границы нормы
            timestamp=datetime.now(),
        )

        result_below = self.engine.analyze(sensor_data_below)

        self.assertIsNotNone(result_below)
        self.assertTrue(result_below["is_anomaly"])  # Ниже границы - аномалия
        self.assertEqual(result_below["anomaly_type"], "low_pressure")

    def test_critical_pressure_anomaly(self):
        """Тест критической аномалии давления"""
        # Создаем данные с критически высоким давлением
        sensor_data = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("295.0"),  # Близко к максимуму (300.0)
            timestamp=datetime.now(),
        )

        result = self.engine.analyze(sensor_data)

        self.assertIsNotNone(result)
        self.assertTrue(result["is_anomaly"])
        self.assertEqual(result["anomaly_type"], "high_pressure")
        self.assertGreater(result["severity"], 5)  # Высокая степень серьезности

    def test_multiple_sensors_analysis(self):
        """Тест анализа нескольких сенсоров"""
        # Создаем второй датчик
        temp_sensor = Sensor.objects.create(
            equipment=self.equipment,
            name="Датчик температуры",
            type="temperature",
            unit="celsius",
            min_value=Decimal("-20.0"),
            max_value=Decimal("150.0"),
            normal_min=Decimal("20.0"),
            normal_max=Decimal("80.0"),
        )

        # Нормальное давление
        pressure_data = SensorData.objects.create(
            sensor=self.pressure_sensor,
            value=Decimal("150.0"),
            timestamp=datetime.now(),
        )

        # Аномальная температура
        temp_data = SensorData.objects.create(
            sensor=temp_sensor,
            value=Decimal("120.0"),  # Выше нормы
            timestamp=datetime.now(),
        )

        pressure_result = self.engine.analyze(pressure_data)
        temp_result = self.engine.analyze(temp_data)

        self.assertFalse(pressure_result["is_anomaly"])
        self.assertTrue(temp_result["is_anomaly"])
        self.assertEqual(temp_result["anomaly_type"], "high_temperature")
