"""Модуль проекта с автогенерированным докстрингом."""

from typing import Any

from django.utils import timezone

from .models import DiagnosticReport, HydraulicSystem


class DiagnosticEngine:
    """Основной движок диагностики гидравлических систем.
    Анализирует данные датчиков, выявляет аномалии и создает диагностические отчёты.
    """

    # Пороговые значения для различных параметров
    ANOMALY_THRESHOLDS: dict[str, dict[str, float]] = {
        "pressure": {"min": 10, "max": 300},  # бар
        "temperature": {"min": 20, "max": 80},  # °C
        "flow_rate": {"min": 0.1, "max": 100},  # л/мин
        "vibration": {"min": 0, "max": 5},  # мм/с
        "oil_level": {"min": 20, "max": 100},  # %
    }

    def analyze_system(
        self, system_id: str, sensor_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Основной метод анализа системы.

        Args:
            system_id: ID гидравлической системы
            sensor_data: Данные от датчиков системы

        Returns:
            Dict с результатами анализа включая аномалии и диагностический отчёт

        """
        try:
            system = HydraulicSystem.objects.get(id=system_id)
            anomalies = self.detect_anomalies(sensor_data)

            report_dict: dict[str, Any] | None = None
            if anomalies:
                report = self.create_report(system, anomalies, sensor_data)
                report_dict = {
                    "id": str(report.id),
                    "title": report.title,
                    "severity": report.severity,
                    "status": report.status,
                    "ai_confidence": report.ai_confidence,
                    "created_at": report.created_at,
                }

            return {
                "system_id": system_id,
                "timestamp": timezone.now(),
                "sensor_data": sensor_data,
                "anomalies": anomalies,
                "report": report_dict,
                "status": (
                    "critical"
                    if any(a.get("severity") == "critical" for a in anomalies)
                    else "normal"
                ),
            }

        except HydraulicSystem.DoesNotExist as exc:
            raise ValueError(f"Система с ID {system_id} не найдена") from exc
        except Exception as exc:
            raise RuntimeError(f"Ошибка при анализе системы: {exc}") from exc

    def detect_anomalies(self, sensor_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Выявляет аномалии в данных датчиков."""
        anomalies: list[dict[str, Any]] = []

        for parameter, value in sensor_data.items():
            if self.is_anomaly(parameter, float(value)):
                severity = self._calculate_severity(parameter, float(value))
                anomalies.append(
                    {
                        "parameter": parameter,
                        "value": float(value),
                        "threshold": self.ANOMALY_THRESHOLDS.get(parameter, {}),
                        "severity": severity,
                        "timestamp": timezone.now(),
                        "message": self._get_anomaly_message(
                            parameter, float(value), severity
                        ),
                    }
                )

        return anomalies

    def is_anomaly(self, parameter: str, value: float) -> bool:
        """Проверяет, является ли значение параметра аномальным."""
        if parameter not in self.ANOMALY_THRESHOLDS:
            return False
        thresholds = self.ANOMALY_THRESHOLDS[parameter]
        return value < thresholds.get("min", float("-inf")) or value > thresholds.get(
            "max", float("inf")
        )

    def create_report(
        self,
        system: HydraulicSystem,
        anomalies: list[dict[str, Any]],
        _sensor_data: dict[str, Any],
    ) -> DiagnosticReport:
        """Создаёт диагностический отчёт на основе аномалий."""
        has_critical = any(a.get("severity") == "critical" for a in anomalies)
        severity = "critical" if has_critical else "warning"

        title = "Критические аномалии" if has_critical else "Предупреждения"
        description = self.format_anomalies_description(anomalies)

        return DiagnosticReport.objects.create(
            system=system,
            title=title,
            severity=severity,
            status="open",
            ai_confidence=0.8 if has_critical else 0.5,
            impacted_components_count=0,
            description=description,
            created_at=timezone.now(),
        )

    def format_anomalies_description(self, anomalies: list[dict[str, Any]]) -> str:
        """Форматирует описание аномалий в читаемый текст."""
        if not anomalies:
            return "Аномалий не обнаружено"
        lines: list[str] = []
        for anomaly in anomalies:
            parameter = anomaly["parameter"]
            value = anomaly["value"]
            severity = anomaly["severity"]
            message = anomaly.get("message", "")
            line = f"Параметр '{parameter}': {value} ({severity.upper()})"
            if message:
                line += f" - {message}"
            lines.append(line)
        return "\n".join(lines)

    def _calculate_severity(self, parameter: str, value: float) -> str:
        """Вычисляет степень критичности аномалии."""
        thresholds = self.ANOMALY_THRESHOLDS.get(parameter, {})
        min_threshold = thresholds.get("min", float("-inf"))
        max_threshold = thresholds.get("max", float("inf"))
        if parameter == "pressure" and (
            value < min_threshold * 0.5 or value > max_threshold * 1.5
        ):
            return "critical"
        if parameter == "temperature" and value > max_threshold * 1.2:
            return "critical"
        if parameter == "oil_level" and value < min_threshold * 0.5:
            return "critical"
        return "warning"

    def _get_anomaly_message(self, parameter: str, value: float, _severity: str) -> str:
        """Генерирует понятное сообщение об аномалии."""
        messages: dict[str, dict[str, str]] = {
            "pressure": {
                "low": "Давление ниже нормы, возможна утечка",
                "high": "Давление выше нормы, риск повреждения системы",
            },
            "temperature": {
                "low": "Температура ниже нормы",
                "high": "Перегрев системы, требуется охлаждение",
            },
            "flow_rate": {
                "low": "Низкая скорость потока, возможна блокировка",
                "high": "Высокая скорость потока",
            },
            "vibration": {
                "low": "Низкая вибрация",
                "high": "Повышенная вибрация, проверьте подшипники",
            },
            "oil_level": {
                "low": "Низкий уровень масла, требуется доливка",
                "high": "Высокий уровень масла",
            },
        }
        if parameter not in messages:
            return f"Аномальное значение: {value}"
        thresholds = self.ANOMALY_THRESHOLDS.get(parameter, {})
        if value < thresholds.get("min", float("-inf")):
            return messages[parameter].get("low", f"Значение {value} ниже нормы")
        if value > thresholds.get("max", float("inf")):
            return messages[parameter].get("high", f"Значение {value} выше нормы")
        return f"Аномальное значение: {value}"
