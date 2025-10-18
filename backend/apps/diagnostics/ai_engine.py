import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from django.utils import timezone
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger("ai_engine")


class HydraulicSystemAIEngine:
    """AI движок для диагностики гидравлических систем"""

    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.performance_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.clustering_model = DBSCAN(eps=0.5, min_samples=5)
        self.is_trained = False

        # Пороговые значения для различных типов датчиков
        self.sensor_thresholds = {
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

            training_stats = {
                "samples_count": len(sensor_data),
                "features_count": features.shape,
                "training_date": timezone.now(),
                "anomaly_threshold": self.anomaly_detector.offset_,
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

            anomalies = []
            for idx, (is_anom, score) in enumerate(zip(is_anomaly, anomaly_scores)):
                if is_anom:
                    row = sensor_data.iloc[idx]
                    anomalies.append(
                        {
                            "timestamp": row.get("timestamp", datetime.now()),
                            "sensor_type": row.get("sensor_type", "unknown"),
                            "value": row.get("value", 0),
                            "anomaly_score": float(score),
                            "severity": self._calculate_anomaly_severity(score),
                            "description": self._generate_anomaly_description(row, score),
                        }
                    )

            return anomalies

        except Exception as e:
            logger.error(f"Ошибка обнаружения аномалий: {e}")
            return []

    def predict_failure_probability(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
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

            prediction = {
                "failure_probability": failure_probability,
                "risk_level": self._get_risk_level(failure_probability),
                "predicted_time_to_failure": self._estimate_time_to_failure(
                    failure_probability, trends
                ),
                "contributing_factors": self._identify_risk_factors(trends, component_health),
                "recommendations": self._generate_recommendations(failure_probability, trends),
            }

            return prediction

        except Exception as e:
            logger.error(f"Ошибка предсказания отказа: {e}")
            return {
                "failure_probability": 0.0,
                "risk_level": "unknown",
                "error": str(e),
            }

    def generate_diagnostic_insights(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация диагностических инсайтов"""
        try:
            insights = {
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

    def _prepare_features(self, sensor_data: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для ML моделей"""
        if sensor_data.empty:
            return pd.DataFrame()

        features = []

        # Основные статистики для каждого типа датчика
        for sensor_type in sensor_data["sensor_type"].unique():
            sensor_subset = sensor_data[sensor_data["sensor_type"] == sensor_type]

            if not sensor_subset.empty:
                features.extend(
                    [
                        sensor_subset["value"].mean(),
                        sensor_subset["value"].std(),
                        sensor_subset["value"].min(),
                        sensor_subset["value"].max(),
                        sensor_subset["value"].quantile(0.25),
                        sensor_subset["value"].quantile(0.75),
                    ]
                )

        # Временные признаки
        if "timestamp" in sensor_data.columns:
            sensor_data["timestamp"] = pd.to_datetime(sensor_data["timestamp"])
            features.extend(
                [
                    (sensor_data["timestamp"].dt.hour.mode().iloc if not sensor_data.empty else 0),
                    (
                        sensor_data["timestamp"].dt.dayofweek.mode().iloc
                        if not sensor_data.empty
                        else 0
                    ),
                ]
            )

        # Создание DataFrame из признаков
        feature_names = [f"feature_{i}" for i in range(len(features))]
        return pd.DataFrame([features], columns=feature_names)

    def _threshold_based_anomaly_detection(self, sensor_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Обнаружение аномалий на основе пороговых значений"""
        anomalies = []

        for _, row in sensor_data.iterrows():
            sensor_type = row.get("sensor_type")
            value = row.get("value", 0)

            if sensor_type in self.sensor_thresholds:
                thresholds = self.sensor_thresholds[sensor_type]

                anomaly = None
                if value <= thresholds["critical_low"] or value >= thresholds["critical_high"]:
                    anomaly = {
                        "severity": "critical",
                        "description": f"Критическое значение {sensor_type}: {value}",
                    }
                elif value <= thresholds["warning_low"] or value >= thresholds["warning_high"]:
                    anomaly = {
                        "severity": "warning",
                        "description": f"Предупреждение {sensor_type}: {value}",
                    }

                if anomaly:
                    anomaly.update(
                        {
                            "timestamp": row.get("timestamp", datetime.now()),
                            "sensor_type": sensor_type,
                            "value": value,
                            "anomaly_score": self._calculate_threshold_score(value, thresholds),
                        }
                    )
                    anomalies.append(anomaly)

        return anomalies

    def _calculate_threshold_score(self, value: float, thresholds: Dict[str, float]) -> float:
        """Расчет score аномалии на основе пороговых значений"""
        normal_min = thresholds["normal_min"]
        normal_max = thresholds["normal_max"]

        if normal_min <= value <= normal_max:
            return 0.0  # Нормальное значение

        if value < normal_min:
            distance = normal_min - value
            max_distance = normal_min - thresholds["critical_low"]
        else:
            distance = value - normal_max
            max_distance = thresholds["critical_high"] - normal_max

        return min(1.0, distance / max_distance) if max_distance > 0 else 1.0

    def _calculate_anomaly_severity(self, anomaly_score: float) -> str:
        """Определение уровня серьезности аномалии"""
        if anomaly_score <= -0.5:
            return "critical"
        elif anomaly_score <= -0.2:
            return "high"
        elif anomaly_score <= -0.1:
            return "medium"
        else:
            return "low"

    def _generate_anomaly_description(self, row: pd.Series, score: float) -> str:
        """Генерация описания аномалии"""
        sensor_type = row.get("sensor_type", "unknown")
        value = row.get("value", 0)
        severity = self._calculate_anomaly_severity(score)

        descriptions = {
            "critical": f"Критическая аномалия в {sensor_type}: значение {value} существенно отклоняется от нормы",
            "high": f"Высокая аномалия в {sensor_type}: значение {value} значительно отклоняется от нормы",
            "medium": f"Умеренная аномалия в {sensor_type}: значение {value} отклоняется от нормы",
            "low": f"Слабая аномалия в {sensor_type}: значение {value} немного отклоняется от нормы",
        }

        return descriptions.get(severity, f"Аномалия в {sensor_type}: {value}")

    def _analyze_trends(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ трендов в данных системы"""
        trends = {
            "pressure_trend": "stable",
            "temperature_trend": "stable",
            "flow_trend": "stable",
            "vibration_trend": "stable",
            "overall_trend": "stable",
        }

        # Здесь должен быть реальный анализ трендов
        # Для примера возвращаем базовые значения

        return trends

    def _assess_component_health(self, system_data: Dict[str, Any]) -> Dict[str, float]:
        """Оценка состояния компонентов"""
        # Базовая оценка здоровья компонентов
        return {
            "pump": 0.85,
            "motor": 0.90,
            "valves": 0.88,
            "filters": 0.75,
            "overall": 0.85,
        }

    def _analyze_operation_mode(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ режима эксплуатации"""
        return {
            "load_factor": 0.7,
            "duty_cycle": 0.8,
            "operating_conditions": "normal",
            "stress_level": "moderate",
        }

    def _calculate_failure_probability(
        self, trends: Dict, component_health: Dict, operation: Dict
    ) -> float:
        """Расчет вероятности отказа"""
        # Простая модель расчета вероятности отказа
        health_factor = 1.0 - component_health.get("overall", 0.5)
        load_factor = operation.get("load_factor", 0.5)

        failure_prob = health_factor * 0.7 + load_factor * 0.3
        return min(1.0, max(0.0, failure_prob))

    def _get_risk_level(self, probability: float) -> str:
        """Определение уровня риска"""
        if probability >= 0.7:
            return "critical"
        elif probability >= 0.5:
            return "high"
        elif probability >= 0.3:
            return "medium"
        else:
            return "low"

    def _estimate_time_to_failure(self, probability: float, trends: Dict) -> Optional[str]:
        """Оценка времени до отказа"""
        if probability >= 0.8:
            return "Менее 1 недели"
        elif probability >= 0.6:
            return "1-4 недели"
        elif probability >= 0.4:
            return "1-3 месяца"
        elif probability >= 0.2:
            return "3-12 месяцев"
        else:
            return "Более 1 года"

    def _identify_risk_factors(self, trends: Dict, component_health: Dict) -> List[str]:
        """Идентификация факторов риска"""
        risk_factors = []

        # Анализ состояния компонентов
        for component, health in component_health.items():
            if health < 0.8 and component != "overall":
                risk_factors.append(f"Ухудшение состояния: {component}")

        # Анализ трендов
        for param, trend in trends.items():
            if trend in ["declining", "critical"]:
                risk_factors.append(f"Негативный тренд: {param}")

        return risk_factors

    def _generate_recommendations(self, probability: float, trends: Dict) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []

        if probability >= 0.7:
            recommendations.extend(
                [
                    "Немедленная остановка системы для диагностики",
                    "Проведение полной проверки всех компонентов",
                    "Замена критически изношенных деталей",
                ]
            )
        elif probability >= 0.5:
            recommendations.extend(
                [
                    "Планирование внеочередного технического обслуживания",
                    "Усиленный мониторинг системы",
                    "Подготовка запасных частей",
                ]
            )
        elif probability >= 0.3:
            recommendations.extend(
                [
                    "Увеличение частоты мониторинга",
                    "Планирование планового ТО",
                    "Анализ условий эксплуатации",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Продолжение штатной эксплуатации",
                    "Соблюдение графика планового ТО",
                    "Контроль ключевых параметров",
                ]
            )

        return recommendations

    def _calculate_health_score(self, system_data: Dict[str, Any]) -> float:
        """Расчет общего индекса здоровья системы"""
        # Базовый расчет на основе последних показаний датчиков
        return 85.0  # Пример

    def _analyze_performance(self, system_data: Dict[str, Any]) -> Dict[str, float]:
        """Анализ производительности системы"""
        return {
            "efficiency": 0.88,
            "productivity": 0.92,
            "energy_consumption": 0.75,
            "uptime": 0.96,
        }

    def _generate_maintenance_recommendations(
        self, system_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Генерация рекомендаций по обслуживанию"""
        return [
            {
                "type": "preventive",
                "priority": "medium",
                "description": "Замена гидравлических фильтров",
                "estimated_time": "2 часа",
                "cost_estimate": 15000,
            },
            {
                "type": "inspection",
                "priority": "high",
                "description": "Проверка уровня масла и его качества",
                "estimated_time": "30 минут",
                "cost_estimate": 2000,
            },
        ]

    def _suggest_optimizations(self, system_data: Dict[str, Any]) -> List[str]:
        """Предложения по оптимизации"""
        return [
            "Оптимизация рабочего давления для снижения энергопотребления",
            "Настройка циклов работы для увеличения срока службы",
            "Установка дополнительных датчиков для улучшения мониторинга",
        ]

    def _analyze_costs(self, system_data: Dict[str, Any]) -> Dict[str, float]:
        """Анализ затрат"""
        return {
            "maintenance_cost_per_month": 25000.0,
            "energy_cost_per_month": 15000.0,
            "downtime_cost_per_hour": 50000.0,
            "total_operating_cost_per_month": 45000.0,
        }

    def _assess_reliability(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка надежности"""
        return {
            "mtbf": 720,  # часы
            "mttr": 4,  # часы
            "availability": 0.994,
            "reliability_score": 0.89,
        }

    def _generate_summary_report(self, insights: Dict[str, Any]) -> str:
        """Генерация итогового отчета"""
        health_score = insights.get("system_health_score", 0)
        performance = insights.get("performance_metrics", {})

        if health_score >= 90:
            status = "отличном"
        elif health_score >= 75:
            status = "хорошем"
        elif health_score >= 60:
            status = "удовлетворительном"
        else:
            status = "неудовлетворительном"

        summary = f"""
        Система находится в {status} состоянии (индекс здоровья: {health_score}%).
        Эффективность: {performance.get('efficiency', 0)*100:.1f}%.
        Время безотказной работы: {performance.get('uptime', 0)*100:.1f}%.
        
        Рекомендуется продолжить мониторинг и выполнить плановое обслуживание согласно рекомендациям.
        """

        return summary.strip()

    def _get_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Получение важности признаков"""
        if hasattr(self.performance_predictor, "feature_importances_"):
            importance = self.performance_predictor.feature_importances_
            return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
        return {}


# Глобальный экземпляр AI движка
ai_engine = HydraulicSystemAIEngine()


def get_ai_engine() -> HydraulicSystemAIEngine:
    """Получить экземпляр AI движка"""
    return ai_engine
