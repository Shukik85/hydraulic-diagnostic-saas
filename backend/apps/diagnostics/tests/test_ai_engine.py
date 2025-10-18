import unittest
from datetime import datetime, timedelta

import pandas as pd
from apps.diagnostics.ai_engine import ai_engine
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
            location="Тестовый цех",
            status="active",
            max_pressure=250.0,
            flow_rate=75.0,
            temperature_range="-10°C до +80°C",
            owner=self.user,
        )

    def create_test_sensor_data(self, hours_back=24, interval_minutes=30):
        """Создание тестовых данных датчиков"""
        sensor_data = []
        start_time = datetime.now() - timedelta(hours=hours_back)

        # Генерация данных с интервалом
        current_time = start_time
        while current_time <= datetime.now():
            # Нормальные значения
            sensor_data.extend(
                [
                    SensorData.objects.create(
                        system=self.system,
                        sensor_type="pressure",
                        value=200.0 + (current_time.hour - 12) * 5,  # Дневные колебания
                        unit="bar",
                        timestamp=current_time,
                        is_critical=False,
                    ),
                    SensorData.objects.create(
                        system=self.system,
                        sensor_type="temperature",
                        value=50.0 + (current_time.hour - 12) * 2,
                        unit="°C",
                        timestamp=current_time,
                        is_critical=False,
                    ),
                    SensorData.objects.create(
                        system=self.system,
                        sensor_type="flow",
                        value=70.0 + (current_time.hour - 12) * 1,
                        unit="л/мин",
                        timestamp=current_time,
                        is_critical=False,
                    ),
                    SensorData.objects.create(
                        system=self.system,
                        sensor_type="vibration",
                        value=5.0 + abs((current_time.hour - 12) * 0.5),
                        unit="мм/с",
                        timestamp=current_time,
                        is_critical=False,
                    ),
                ]
            )

            current_time += timedelta(minutes=interval_minutes)

        return sensor_data

    def test_feature_extraction(self):
        """Тест извлечения признаков"""
        # Создание тестовых данных
        self.create_test_sensor_data(hours_back=2)

        # Получение данных для анализа
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

        # Извлечение признаков
        features = ai_engine.extract_features(df)

        # Проверки
        self.assertIsInstance(features, pd.DataFrame)
        self.assertGreater(len(features.columns), 0)
        self.assertIn("pressure", features.columns)
        self.assertIn("temperature", features.columns)
        self.assertIn("pressure_variance", features.columns)
        self.assertIn("pressure_trend", features.columns)

    def test_anomaly_detection(self):
        """Тест обнаружения аномалий"""
        # Создание данных с аномалиями
        self.create_test_sensor_data(hours_back=1)

        # Добавление критических значений
        SensorData.objects.create(
            system=self.system,
            sensor_type="pressure",
            value=450.0,  # Критически высокое давление
            unit="bar",
            timestamp=datetime.now(),
            is_critical=True,
            warning_message="Критическое давление!",
        )

        sensor_data = self.system.sensor_data.all()
        df = pd.DataFrame(
            [
                {
                    "sensor_type": data.sensor_type,
                    "value": data.value,
                    "timestamp": data.timestamp,
                    "is_critical": data.is_critical,
                }
                for data in sensor_data
            ]
        )

        features = ai_engine.extract_features(df)
        anomalies = ai_engine.detect_anomalies(features)

        # Проверки
        self.assertIsInstance(anomalies, dict)
        self.assertIn("anomaly_score", anomalies)
        self.assertIn("anomalies", anomalies)
        self.assertIn("is_anomalous", anomalies)
        self.assertTrue(anomalies["is_anomalous"])  # Должна быть обнаружена аномалия
        self.assertGreater(len(anomalies["anomalies"]), 0)

    def test_failure_prediction(self):
        """Тест предсказания отказов"""
        # Создание данных с трендом к отказу
        self.create_test_sensor_data(hours_back=2)

        # Добавление данных с тревожными трендами
        for i in range(5):
            SensorData.objects.create(
                system=self.system,
                sensor_type="pressure",
                value=300.0 + i * 20,  # Растущее давление
                unit="bar",
                timestamp=datetime.now() - timedelta(minutes=i * 10),
                is_critical=i >= 2,
            )

        sensor_data = self.system.sensor_data.all()
        df = pd.DataFrame(
            [
                {
                    "sensor_type": data.sensor_type,
                    "value": data.value,
                    "timestamp": data.timestamp,
                    "is_critical": data.is_critical,
                }
                for data in sensor_data
            ]
        )

        features = ai_engine.extract_features(df)
        predictions = ai_engine.predict_failure(features, df)

        # Проверки
        self.assertIsInstance(predictions, dict)
        self.assertIn("failure_probability", predictions)
        self.assertIn("predictions", predictions)
        self.assertIn("maintenance_urgency", predictions)
        self.assertGreater(predictions["failure_probability"], 0)

    def test_comprehensive_analysis(self):
        """Тест комплексного анализа"""
        # Создание разнообразных данных
        self.create_test_sensor_data(hours_back=3)

        # Добавление критических событий
        critical_data = [
            {"sensor_type": "pressure", "value": 400.0, "is_critical": True},
            {"sensor_type": "temperature", "value": 90.0, "is_critical": True},
            {"sensor_type": "vibration", "value": 30.0, "is_critical": True},
        ]

        for data in critical_data:
            SensorData.objects.create(
                system=self.system,
                timestamp=datetime.now(),
                unit="test",
                warning_message="Test critical event",
                **data
            )

        sensor_data = self.system.sensor_data.all()
        df = pd.DataFrame(
            [
                {
                    "sensor_type": data.sensor_type,
                    "value": data.value,
                    "timestamp": data.timestamp,
                    "is_critical": data.is_critical,
                    "warning_message": data.warning_message,
                }
                for data in sensor_data
            ]
        )

        system_info = {
            "id": self.system.id,
            "name": self.system.name,
            "status": self.system.status,
            "max_pressure": self.system.max_pressure,
            "flow_rate": self.system.flow_rate,
        }

        # Выполнение комплексного анализа
        analysis = ai_engine.perform_comprehensive_analysis(df, system_info)

        # Проверки результата
        self.assertIsInstance(analysis, dict)

        # Проверка основных разделов
        required_keys = [
            "timestamp",
            "system_health",
            "anomalies",
            "predictions",
            "recommendations",
            "summary",
        ]

        for key in required_keys:
            self.assertIn(key, analysis)

        # Проверка структуры system_health
        system_health = analysis["system_health"]
        self.assertIn("score", system_health)
        self.assertIn("status", system_health)
        self.assertIsInstance(system_health["score"], (int, float))

        # Проверка аномалий
        anomalies = analysis["anomalies"]
        self.assertIn("anomaly_score", anomalies)
        self.assertIn("is_anomalous", anomalies)

        # Проверка рекомендаций
        recommendations = analysis["recommendations"]
        self.assertIsInstance(recommendations, list)
        if recommendations:
            self.assertIn("priority", recommendations)
            self.assertIn("title", recommendations)
            self.assertIn("action", recommendations)

    def test_empty_data_handling(self):
        """Тест обработки пустых данных"""
        empty_df = pd.DataFrame()

        # Тест извлечения признаков с пустыми данными
        features = ai_engine.extract_features(empty_df)
        self.assertIsInstance(features, pd.DataFrame)

        # Тест анализа с пустыми данными
        system_info = {"id": 1, "name": "Test", "status": "active"}
        analysis = ai_engine.perform_comprehensive_analysis(empty_df, system_info)

        self.assertIn("system_health", analysis)
        self.assertEqual(analysis["system_health"]["score"], 0)


class RAGSystemTestCase(TestCase):
    """Тесты для RAG системы"""

    def test_knowledge_search(self):
        """Тест поиска в базе знаний"""
        from apps.diagnostics.rag_system import rag_system

        # Поиск по ключевым словам
        results = rag_system.search_knowledge("давление гидросистема", top_k=3)

        self.assertIsInstance(results, list)
        if results:
            self.assertIn("title", results)
            self.assertIn("content", results)
            self.assertIn("relevance_score", results)

    def test_contextual_answer_generation(self):
        """Тест генерации контекстных ответов"""
        from apps.diagnostics.rag_system import rag_system

        question = "Какие причины высокого давления в гидросистеме?"
        answer = rag_system.generate_contextual_answer(question)

        self.assertIsInstance(answer, dict)
        self.assertIn("answer", answer)
        self.assertIn("confidence", answer)
        self.assertIn("sources", answer)

        if (
            answer["answer"]
            != "К сожалению, по данному вопросу информация в базе знаний не найдена."
        ):
            self.assertGreater(len(answer["answer"]), 10)


if __name__ == "__main__":
    unittest.main()
