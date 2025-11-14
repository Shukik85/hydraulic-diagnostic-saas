"""A/B Testing Framework"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

from .versioning import model_registry, ModelVersion

logger = logging.getLogger(__name__)

@dataclass
class ABTestConfig:
    """Конфигурация A/B теста"""
    name: str
    model_type: str
    control_version: str
    treatment_version: str
    treatment_traffic_pct: int = 10
    start_date: datetime = None
    duration_days: int = 7
    min_requests: int = 1000
    max_error_rate_increase: float = 0.05
    min_confidence_improvement: float = 0.02

class ABTestManager:
    """Менеджер A/B тестов"""
    
    def __init__(self):
        self._active_tests: Dict[str, ABTestConfig] = {}
        self._test_results: Dict[str, Dict] = {}
    
    def start_test(self, config: ABTestConfig):
        """Запустить A/B тест"""
        versions = model_registry.list_versions(config.model_type)
        version_names = [v.version for v in versions]
        
        if config.control_version not in version_names:
            raise ValueError(f"Control version {config.control_version} not found")
        if config.treatment_version not in version_names:
            raise ValueError(f"Treatment version {config.treatment_version} not found")
        
        for v in versions:
            if v.version == config.control_version:
                v.traffic_percentage = 100 - config.treatment_traffic_pct
            elif v.version == config.treatment_version:
                v.traffic_percentage = config.treatment_traffic_pct
            else:
                v.traffic_percentage = 0
        
        config.start_date = datetime.utcnow()
        self._active_tests[config.name] = config
        logger.info(f"Started A/B test '{config.name}': {config.control_version} vs {config.treatment_version}")
    
    def assign_variant(self, model_type: str, user_id: str = None) -> Dict:
        """Получить вариант модели для пользователя"""
        active_test = None
        for test in self._active_tests.values():
            if test.model_type == model_type:
                active_test = test
                break
        
        model_version = model_registry.get_version_for_request(model_type, user_id)
        
        if active_test:
            variant_name = 'control' if model_version.version == active_test.control_version else 'treatment'
        else:
            variant_name = 'champion'
        
        return {
            'name': variant_name,
            'version': model_version.version,
            'model': model_version
        }
    
    def record_result(
        self,
        test_name: str,
        variant: str,
        inference_time_ms: float,
        confidence: float,
        error: bool = False
    ):
        """Записать результат запроса"""
        if test_name not in self._test_results:
            self._test_results[test_name] = {
                'control': {'requests': 0, 'errors': 0, 'total_time': 0, 'total_confidence': 0},
                'treatment': {'requests': 0, 'errors': 0, 'total_time': 0, 'total_confidence': 0}
            }
        
        results = self._test_results[test_name][variant]
        results['requests'] += 1
        results['total_time'] += inference_time_ms
        results['total_confidence'] += confidence
        if error:
            results['errors'] += 1
    
    def evaluate_test(self, test_name: str) -> Dict:
        """Оценить результаты A/B теста"""
        if test_name not in self._active_tests:
            raise ValueError(f"Test {test_name} not found")
        
        config = self._active_tests[test_name]
        results = self._test_results.get(test_name, {})
        control = results.get('control', {})
        treatment = results.get('treatment', {})
        
        if treatment.get('requests', 0) < config.min_requests:
            return {
                'decision': 'continue',
                'metrics': results,
                'recommendation': f"Недостаточно данных ({treatment.get('requests', 0)}/{config.min_requests})"
            }
        
        control_error_rate = control['errors'] / control['requests'] if control['requests'] > 0 else 0
        treatment_error_rate = treatment['errors'] / treatment['requests'] if treatment['requests'] > 0 else 0
        control_avg_confidence = control['total_confidence'] / control['requests'] if control['requests'] > 0 else 0
        treatment_avg_confidence = treatment['total_confidence'] / treatment['requests'] if treatment['requests'] > 0 else 0
        
        error_increase = treatment_error_rate - control_error_rate
        confidence_improvement = treatment_avg_confidence - control_avg_confidence
        
        if error_increase > config.max_error_rate_increase:
            decision = 'rollback'
            recommendation = f"Ошибок больше на {error_increase*100:.2f}%. Откатываем treatment."
        elif confidence_improvement >= config.min_confidence_improvement:
            decision = 'promote'
            recommendation = f"Confidence улучшился на {confidence_improvement*100:.2f}%. Промоутим treatment."
        else:
            decision = 'continue'
            recommendation = "Результаты неоднозначны. Продолжаем тест."
        
        return {
            'decision': decision,
            'metrics': {
                'control': {
                    'requests': control['requests'],
                    'error_rate': control_error_rate,
                    'avg_confidence': control_avg_confidence
                },
                'treatment': {
                    'requests': treatment['requests'],
                    'error_rate': treatment_error_rate,
                    'avg_confidence': treatment_avg_confidence
                }
            },
            'recommendation': recommendation
        }
    
    def finalize_test(self, test_name: str, decision: str):
        """Завершить A/B тест"""
        if test_name not in self._active_tests:
            raise ValueError(f"Test {test_name} not found")
        
        config = self._active_tests[test_name]
        
        if decision == 'promote':
            model_registry.promote_to_champion(config.model_type, config.treatment_version)
            logger.info(f"Test '{test_name}': promoted {config.treatment_version}")
        elif decision == 'rollback':
            versions = model_registry.list_versions(config.model_type)
            for v in versions:
                if v.version == config.control_version:
                    v.traffic_percentage = 100
                else:
                    v.traffic_percentage = 0
            logger.info(f"Test '{test_name}': rolled back to {config.control_version}")
        
        del self._active_tests[test_name]

# Singleton
ab_test_manager = ABTestManager()
