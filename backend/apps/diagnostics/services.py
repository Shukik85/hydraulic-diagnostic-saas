from typing import Dict, List, Any, Optional
from datetime import datetime
from django.utils import timezone
from .models import Diagnosis, HydraulicSystem


class DiagnosticEngine:
    """
    Основной движок диагностики гидравлических систем.
    Анализирует данные датчиков, выявляет аномалии и создает диагнозы.
    """
    
    # Пороговые значения для различных параметров
    ANOMALY_THRESHOLDS = {
        'pressure': {'min': 10, 'max': 300},  # бар
        'temperature': {'min': 20, 'max': 80},  # градусы Цельсия
        'flow_rate': {'min': 0.1, 'max': 100},  # л/мин
        'vibration': {'min': 0, 'max': 5},  # мм/с
        'oil_level': {'min': 20, 'max': 100},  # %
    }
    
    def analyze_system(self, system_id: int, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод анализа системы.
        
        Args:
            system_id: ID гидравлической системы
            sensor_data: Данные от датчиков системы
            
        Returns:
            Dict с результатами анализа включая аномалии и диагноз
        """
        try:
            # Получаем систему из БД
            system = HydraulicSystem.objects.get(id=system_id)
            
            # Выявляем аномалии
            anomalies = self.detect_anomalies(sensor_data)
            
            # Создаем диагноз если есть аномалии
            diagnosis = None
            if anomalies:
                diagnosis = self.create_diagnosis(system, anomalies, sensor_data)
                
                # Проверяем критичность и уведомляем
                self.notify_if_critical(diagnosis)
            
            return {
                'system_id': system_id,
                'timestamp': timezone.now(),
                'sensor_data': sensor_data,
                'anomalies': anomalies,
                'diagnosis': diagnosis.to_dict() if diagnosis else None,
                'status': 'critical' if any(a.get('severity') == 'critical' for a in anomalies) else 'normal'
            }
            
        except HydraulicSystem.DoesNotExist:
            raise ValueError(f"Система с ID {system_id} не найдена")
        except Exception as e:
            raise RuntimeError(f"Ошибка при анализе системы: {str(e)}")
    
    def detect_anomalies(self, sensor_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Выявляет аномалии в данных датчиков.
        
        Args:
            sensor_data: Данные от датчиков
            
        Returns:
            Список обнаруженных аномалий
        """
        anomalies = []
        
        for parameter, value in sensor_data.items():
            if self.is_anomaly(parameter, value):
                severity = self._calculate_severity(parameter, value)
                anomalies.append({
                    'parameter': parameter,
                    'value': value,
                    'threshold': self.ANOMALY_THRESHOLDS.get(parameter, {}),
                    'severity': severity,
                    'timestamp': timezone.now(),
                    'message': self._get_anomaly_message(parameter, value, severity)
                })
        
        return anomalies
    
    def is_anomaly(self, parameter: str, value: float) -> bool:
        """
        Проверяет, является ли значение параметра аномальным.
        
        Args:
            parameter: Название параметра
            value: Значение параметра
            
        Returns:
            True если значение аномально, False если нормально
        """
        if parameter not in self.ANOMALY_THRESHOLDS:
            return False
            
        thresholds = self.ANOMALY_THRESHOLDS[parameter]
        min_threshold = thresholds.get('min', float('-inf'))
        max_threshold = thresholds.get('max', float('inf'))
        
        return value < min_threshold or value > max_threshold
    
    def create_diagnosis(self, system: HydraulicSystem, anomalies: List[Dict[str, Any]], 
                        sensor_data: Dict[str, Any]) -> Diagnosis:
        """
        Создает диагноз на основе обнаруженных аномалий.
        
        Args:
            system: Гидравлическая система
            anomalies: Список аномалий
            sensor_data: Исходные данные датчиков
            
        Returns:
            Созданный объект диагноза
        """
        # Определяем общую критичность
        has_critical = any(a.get('severity') == 'critical' for a in anomalies)
        overall_severity = 'critical' if has_critical else 'warning'
        
        # Формируем описание аномалий
        anomalies_description = self.format_anomalies_description(anomalies)
        
        # Генерируем рекомендации
        recommendations = self._generate_recommendations(anomalies)
        
        # Создаем диагноз
        diagnosis = Diagnosis.objects.create(
            system=system,
            severity=overall_severity,
            description=anomalies_description,
            recommendations=recommendations,
            sensor_data=sensor_data,
            created_at=timezone.now()
        )
        
        return diagnosis
    
    def format_anomalies_description(self, anomalies: List[Dict[str, Any]]) -> str:
        """
        Форматирует описание аномалий в читаемый текст.
        
        Args:
            anomalies: Список аномалий
            
        Returns:
            Форматированное описание аномалий
        """
        if not anomalies:
            return "Аномалий не обнаружено"
        
        descriptions = []
        for anomaly in anomalies:
            parameter = anomaly['parameter']
            value = anomaly['value']
            severity = anomaly['severity']
            message = anomaly.get('message', '')
            
            desc = f"Параметр '{parameter}': {value} ({severity.upper()})"
            if message:
                desc += f" - {message}"
            descriptions.append(desc)
        
        return "\n".join(descriptions)
    
    def notify_if_critical(self, diagnosis: Diagnosis) -> None:
        """
        Отправляет уведомления при критических диагнозах.
        
        Args:
            diagnosis: Диагноз для проверки критичности
        """
        if diagnosis.severity == 'critical':
            # TODO: Интегрировать с NotificationService
            # notification_service = NotificationService()
            # notification_service.send_critical_alert(
            #     system=diagnosis.system,
            #     diagnosis=diagnosis,
            #     recipients=diagnosis.system.get_notification_recipients()
            # )
            
            # Пока что просто логируем
            print(f"КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Система {diagnosis.system.name} требует немедленного внимания!")
            print(f"Диагноз: {diagnosis.description}")
            print(f"Рекомендации: {diagnosis.recommendations}")
    
    def _calculate_severity(self, parameter: str, value: float) -> str:
        """
        Вычисляет степень критичности аномалии.
        
        Args:
            parameter: Название параметра
            value: Значение параметра
            
        Returns:
            Степень критичности: 'warning' или 'critical'
        """
        thresholds = self.ANOMALY_THRESHOLDS.get(parameter, {})
        min_threshold = thresholds.get('min', float('-inf'))
        max_threshold = thresholds.get('max', float('inf'))
        
        # Критичные отклонения (более 50% от нормы)
        if parameter == 'pressure':
            if value < min_threshold * 0.5 or value > max_threshold * 1.5:
                return 'critical'
        elif parameter == 'temperature':
            if value > max_threshold * 1.2:  # Перегрев критичен
                return 'critical'
        elif parameter == 'oil_level':
            if value < min_threshold * 0.5:  # Низкий уровень масла критичен
                return 'critical'
        
        return 'warning'
    
    def _get_anomaly_message(self, parameter: str, value: float, severity: str) -> str:
        """
        Генерирует понятное сообщение об аномалии.
        
        Args:
            parameter: Название параметра
            value: Значение параметра
            severity: Степень критичности
            
        Returns:
            Сообщение об аномалии
        """
        messages = {
            'pressure': {
                'low': 'Давление ниже нормы, возможна утечка',
                'high': 'Давление выше нормы, риск повреждения системы'
            },
            'temperature': {
                'low': 'Температура ниже нормы',
                'high': 'Перегрев системы, требуется охлаждение'
            },
            'flow_rate': {
                'low': 'Низкая скорость потока, возможна блокировка',
                'high': 'Высокая скорость потока'
            },
            'vibration': {
                'low': 'Низкая вибрация',
                'high': 'Повышенная вибрация, проверьте подшипники'
            },
            'oil_level': {
                'low': 'Низкий уровень масла, требуется доливка',
                'high': 'Высокий уровень масла'
            }
        }
        
        if parameter not in messages:
            return f"Аномальное значение: {value}"
        
        thresholds = self.ANOMALY_THRESHOLDS.get(parameter, {})
        min_threshold = thresholds.get('min', float('-inf'))
        max_threshold = thresholds.get('max', float('inf'))
        
        if value < min_threshold:
            return messages[parameter].get('low', f"Значение {value} ниже нормы")
        elif value > max_threshold:
            return messages[parameter].get('high', f"Значение {value} выше нормы")
        
        return f"Аномальное значение: {value}"
    
    def _generate_recommendations(self, anomalies: List[Dict[str, Any]]) -> str:
        """
        Генерирует рекомендации на основе обнаруженных аномалий.
        
        Args:
            anomalies: Список аномалий
            
        Returns:
            Строка с рекомендациями
        """
        recommendations = []
        
        for anomaly in anomalies:
            parameter = anomaly['parameter']
            severity = anomaly['severity']
            
            if parameter == 'pressure' and severity == 'critical':
                recommendations.append("Немедленно остановите систему и проверьте на утечки")
            elif parameter == 'temperature' and severity == 'critical':
                recommendations.append("Остановите систему для охлаждения, проверьте систему охлаждения")
            elif parameter == 'oil_level' and severity == 'critical':
                recommendations.append("Долейте гидравлическое масло до нормального уровня")
            elif parameter == 'vibration' and severity == 'critical':
                recommendations.append("Проверьте подшипники и крепления, возможен дисбаланс")
            else:
                recommendations.append(f"Контролируйте параметр '{parameter}' и при необходимости обратитесь к специалисту")
        
        return "\n".join(recommendations) if recommendations else "Система работает в нормальном режиме"
