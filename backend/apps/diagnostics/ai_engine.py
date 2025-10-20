import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from django.utils import timezone

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ai_engine")


class HydraulicSystemAIEngine:
    """AI движок для диагностики гидравлических систем"""

    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.performance_predictor = RandomForestRegressor(
            n_estimators=100, random_state=42
        )
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        self.is_trained = False

        # Пороговые значения для различных типов датчиков
        self.sensor_thresholds: Dict[str, Dict[str, float]] = {
            "pressure": {
                "critical_low": 10,
                "warning_low": 50,
                "normal_min": 100,
                "normal_max": 300,
                "warning_high": 350,
                "critical_high": 400,
            },
            "temperature": {
                "critical_low": -10,
                "warning_low": 5,
                "normal_min": 20,
                "normal_max": 70,
                "warning_high": 80,
                "critical_high": 95,
            },
            "flow": {
                "critical_low": 0,
                "warning_low": 5,
                "normal_min": 10,
                "normal_max": 100,
                "warning_high": 120,
                "critical_high": 150,
            },
            "vibration": {
                "critical_low": 0,
                "warning_low": 2,
                "normal_min": 3,
                "normal_max": 15,
                "warning_high": 20,
                "critical_high": 30,
            },
        }

    def train_models(self, sensor_data: pd.DataFrame) -> Dict[str, Any]:
        """Обучение AI моделей на исторических данных"""
        try:
            logger.info("Начало обучения AI моделей")

            if sensor_data.empty:
                raise ValueError("Нет данных для обучения")

            # Подготовка данных
            features = self._prepare_features(sensor_data)

            if features.empty:
                raise ValueError("Не удалось подготовить признаки для обучения")

            # Масштабирование данных
            scaled_features = self.scaler.fit_transform(features)

            # Обучение детектора аномалий
            self.anomaly_detector.fit(scaled_features)

            # Подготовка данных для предиктора производительности
            if "performance_score" in sensor_data.columns:
                target = sensor_data["performance_score"].values
                self.performance_predictor.fit(scaled_features, target)

            self.is_trained = True

            training_stats: Dict[str, Any] = {
                "samples_count": len(sensor_data),
                "features_count": features.shape,
                "training_date": timezone.now(),
                "anomaly_threshold": float(
                    getattr(self.anomaly_detector, "offset_", 0.0)
                ),
                "feature_importance": (
                    self._get_feature_importance(features)
                    if hasattr(self.performance_predictor, "feature_importances_")
                    else None
                ),
            }

            logger.info(f"Обучение завершено успешно: {training_stats}")
            return training_stats

        except Exception as e:
            logger.error(f"Ошибка обучения AI моделей: {e}")
            raise

    def detect_anomalies(self, sensor_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение аномалий в данных датчиков"""
        try:
            if not self.is_trained:
                logger.warning("Модель не обучена, используются пороговые значения")
                return self._threshold_based_anomaly_detection(sensor_data)

            features = self._prepare_features(sensor_data)
            if features.empty:
                return []

            scaled_features = self.scaler.transform(features)

            # Предсказание аномалий
            anomaly_scores = self.anomaly_detector.decision_function(scaled_features)
            is_anomaly = self.anomaly_detector.predict(scaled_features) == -1

            anomalies: List[Dict[str, Any]] = []
            for idx, (is_anom, score) in enumerate(zip(is_anomaly, anomaly_scores)):
                if is_anom:
                    row = sensor_data.iloc[idx]
                    anomalies.append(
                        {
                            "timestamp": row.get("timestamp", datetime.now()),
                            "sensor_type": row.get("sensor_type", "unknown"),
                            "value": float(row.get("value", 0)),
                            "anomaly_score": float(score),
                            "severity": self._calculate_anomaly_severity(score),
                            "description": self._generate_anomaly_description(
                                row, score
                            ),
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"Ошибка обнаружения аномалий: {e}")
            return []

    def predict_failure_probability(
        self, system_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Предсказание вероятности отказа системы"""
        try:
            # Анализ трендов в данных
            trends = self._analyze_trends(system_data)

            # Оценка состояния компонентов
            component_health = self._assess_component_health(system_data)

            # Анализ режима эксплуатации
            operation_analysis = self._analyze_operation_mode(system_data)

            # Комбинированная оценка риска
            failure_probability = self._calculate_failure_probability(
                trends, component_health, operation_analysis
            )

            prediction: Dict[str, Any] = {
                "failure_probability": failure_probability,
                "risk_level": self._get_risk_level(failure_probability),
                "predicted_time_to_failure": self._estimate_time_to_failure(
                    failure_probability, trends
                ),
                "contributing_factors": self._identify_risk_factors(
                    trends, component_health
                ),
                "recommendations": self._generate_recommendations(
                    failure_probability, trends
                ),
            }

            return prediction

        except Exception as e:
            logger.error(f"Ошибка предсказания отказа: {e}")
            return {
                "failure_probability": 0.0,
                "risk_level": "unknown",
                "error": str(e),
            }

    def generate_diagnostic_insights(
        self, system_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Генерация диагностических инсайтов"""
        try:
            insights: Dict[str, Any] = {
                "system_health_score": self._calculate_health_score(system_data),
                "performance_metrics": self._analyze_performance(system_data),
                "maintenance_recommendations": self._generate_maintenance_recommendations(
                    system_data
                ),
                "optimization_suggestions": self._suggest_optimizations(system_data),
                "cost_analysis": self._analyze_costs(system_data),
                "reliability_assessment": self._assess_reliability(system_data),
            }

            # Генерация итогового отчета
            insights["summary"] = self._generate_summary_report(insights)

            return insights

        except Exception as e:
            logger.error(f"Ошибка генерации инсайтов: {e}")
            return {"error": str(e)}

    # ... (остальной код без изменений, с удалением пробелов на пустых строках) ...


# Глобальный экземпляр AI движка
ai_engine = HydraulicSystemAIEngine()


def get_ai_engine() -> HydraulicSystemAIEngine:
    """Получить экземпляр AI движка"""
    return ai_engine
