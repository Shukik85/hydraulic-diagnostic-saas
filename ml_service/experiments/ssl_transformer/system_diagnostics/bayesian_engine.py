"""
Bayesian Diagnostic Engine
Обновляет вероятности неисправностей на основе наблюдений
"""
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from knowledge_graph import HydraulicSystemKnowledgeGraph, FaultCause


@dataclass
class ComponentObservation:
    component: str
    fault_detected: bool
    confidence: float
    fault_type: Optional[str] = None
    sensor_readings: Optional[Dict] = None


@dataclass
class DiagnosticResult:
    component: str
    fault_probability: float
    confidence: float
    evidence_type: str  # "observed", "inferred"
    reasoning: str
    severity: str
    recommendations: List[str]


class BayesianDiagnosticEngine:
    """
    Байесовский движок для диагностики с частичными данными
    """
    
    def __init__(self):
        self.knowledge_graph = HydraulicSystemKnowledgeGraph()
    
    def diagnose(
        self,
        symptom: str,
        observations: List[ComponentObservation]
    ) -> List[DiagnosticResult]:
        """
        Главный метод диагностики
        
        Args:
            symptom: Наблюдаемый симптом (например, "boom_rotation_weak")
            observations: Результаты диагностики компонентов с датчиками
        
        Returns:
            Список вероятностей неисправностей для ВСЕХ компонентов
        """
        # Получаем информацию о симптоме
        symptom_info = self.knowledge_graph.get_symptom(symptom)
        if not symptom_info:
            raise ValueError(f"Unknown symptom: {symptom}")
        
        # Инициализируем prior probabilities
        beliefs = {
            cause.component: {
                "probability": cause.prior_probability,
                "severity": cause.severity,
                "description": cause.description,
                "symptom": cause.symptom
            }
            for cause in symptom_info.possible_causes
        }
        
        # Создаём карту наблюдений
        obs_map = {obs.component: obs for obs in observations}
        
        # Байесовское обновление на основе наблюдений
        for component, belief in beliefs.items():
            if component in obs_map:
                obs = obs_map[component]
                beliefs[component] = self._update_belief_with_observation(
                    belief, obs
                )
            else:
                beliefs[component] = self._update_belief_without_observation(
                    belief, obs_map, symptom_info
                )
        
        # Нормализация вероятностей
        total_prob = sum(b["probability"] for b in beliefs.values())
        if total_prob > 0:
            for comp in beliefs:
                beliefs[comp]["probability"] /= total_prob
        
        # Формируем результаты
        results = []
        for component, belief in beliefs.items():
            evidence_type = "observed" if component in obs_map else "inferred"
            
            result = DiagnosticResult(
                component=component,
                fault_probability=belief["probability"],
                confidence=belief.get("confidence", 0.5),
                evidence_type=evidence_type,
                reasoning=self._generate_reasoning(
                    component, belief, evidence_type, obs_map
                ),
                severity=belief["severity"],
                recommendations=self._generate_recommendations(
                    component, belief, evidence_type
                )
            )
            results.append(result)
        
        # Сортируем по вероятности
        results.sort(key=lambda x: x.fault_probability, reverse=True)
        
        return results
    
    def _update_belief_with_observation(
        self,
        prior_belief: Dict,
        observation: ComponentObservation
    ) -> Dict:
        """Обновление вероятности на основе прямого наблюдения"""
        updated = prior_belief.copy()
        
        if observation.fault_detected:
            # Компонент показывает неисправность
            # P(fault | observation) ∝ P(observation | fault) * P(fault)
            likelihood = observation.confidence  # ML model confidence
            updated["probability"] = prior_belief["probability"] * likelihood * 3.0
            updated["confidence"] = observation.confidence
            updated["observation"] = "fault_detected"
        else:
            # Компонент в норме
            updated["probability"] = prior_belief["probability"] * (1 - observation.confidence) * 0.3
            updated["confidence"] = observation.confidence
            updated["observation"] = "no_fault"
        
        return updated
    
    def _update_belief_without_observation(
        self,
        prior_belief: Dict,
        observations: Dict[str, ComponentObservation],
        symptom_info
    ) -> Dict:
        """
        Обновление вероятности для ненаблюдаемых компонентов
        Используем корреляции с наблюдаемыми компонентами
        """
        updated = prior_belief.copy()
        
        # Если все наблюдаемые компоненты в норме, повышаем вероятность ненаблюдаемых
        all_observed_ok = all(
            not obs.fault_detected for obs in observations.values()
        )
        
        if all_observed_ok:
            # "Процесс исключения": если все проверенные OK, проблема скорее всего здесь
            updated["probability"] *= 2.0
            updated["confidence"] = 0.6
            updated["reasoning"] = "elimination"
        else:
            # Есть наблюдаемые неисправности, снижаем вероятность ненаблюдаемых
            updated["probability"] *= 0.7
            updated["confidence"] = 0.4
            updated["reasoning"] = "indirect"
        
        return updated
    
    def _generate_reasoning(
        self,
        component: str,
        belief: Dict,
        evidence_type: str,
        observations: Dict
    ) -> str:
        """Генерация объяснения для пользователя"""
        if evidence_type == "observed":
            if belief.get("observation") == "fault_detected":
                return (
                    f"Датчики на {component} показали неисправность: {belief['symptom']}. "
                    f"Уверенность ML модели: {belief['confidence']*100:.1f}%. "
                    f"{belief['description']}"
                )
            else:
                return (
                    f"Датчики на {component} не показали неисправностей. "
                    f"Уверенность: {belief['confidence']*100:.1f}%."
                )
        else:
            if belief.get("reasoning") == "elimination":
                return (
                    f"Датчики на {component} отсутствуют. "
                    f"Вероятность неисправности повышена, т.к. проверенные компоненты в норме. "
                    f"Возможная причина: {belief['symptom']} - {belief['description']}"
                )
            else:
                return (
                    f"Датчики на {component} отсутствуют. "
                    f"Вероятность рассчитана косвенно на основе других компонентов. "
                    f"Возможная причина: {belief['symptom']}"
                )
    
    def _generate_recommendations(
        self,
        component: str,
        belief: Dict,
        evidence_type: str
    ) -> List[str]:
        """Генерация рекомендаций"""
        recommendations = []
        
        if evidence_type == "observed":
            if belief.get("observation") == "fault_detected":
                recommendations.append(
                    f"🔴 НЕМЕДЛЕННО: Проверить {component} на {belief['symptom']}"
                )
                if belief["severity"] == "critical":
                    recommendations.append(
                        f"⚠️ КРИТИЧНО: Остановить работу до устранения неисправности"
                    )
        else:
            if belief["probability"] > 0.3:
                recommendations.append(
                    f"🟡 РЕКОМЕНДУЕТСЯ: Установить датчики на {component} для точной диагностики"
                )
                recommendations.append(
                    f"📋 Провести визуальный осмотр {component}"
                )
        
        return recommendations


if __name__ == "__main__":
    # Тестовый пример
    engine = BayesianDiagnosticEngine()
    
    # Симптом: слабое усилие поворота стрелы
    # Датчики: только на насосе и моторе
    observations = [
        ComponentObservation(
            component="pump",
            fault_detected=False,
            confidence=0.92,
            sensor_readings={"pressure": 180, "temp": 65}
        ),
        ComponentObservation(
            component="swing_motor",
            fault_detected=True,
            confidence=0.87,
            fault_type="high_friction",
            sensor_readings={"temp": 80, "vibration": 5.2}
        )
    ]
    
    results = engine.diagnose("boom_rotation_weak", observations)
    
    print("🔍 ДИАГНОСТИЧЕСКИЙ ОТЧЁТ")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.component.upper()}")
        print(f"   Вероятность неисправности: {result.fault_probability*100:.1f}%")
        print(f"   Уверенность: {result.confidence*100:.1f}%")
        print(f"   Тип: {result.evidence_type}")
        print(f"   Критичность: {result.severity}")
        print(f"   Объяснение: {result.reasoning}")
        if result.recommendations:
            print(f"   Рекомендации:")
            for rec in result.recommendations:
                print(f"     - {rec}")
