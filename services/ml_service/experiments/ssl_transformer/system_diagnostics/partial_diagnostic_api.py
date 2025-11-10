"""
Production Diagnostic API
Основной интерфейс для частичной диагностики
"""
from typing import Dict, List, Optional
import torch
from pathlib import Path
from bayesian_engine import BayesianDiagnosticEngine, ComponentObservation


class PartialDiagnosticSystem:
    """
    Production система для диагностики с частичными данными
    """
    
    def __init__(self, models_dir: str = "../../checkpoints"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.bayesian_engine = BayesianDiagnosticEngine()
        
        # Загружаем все доступные модели
        self._load_models()
    
    def _load_models(self):
        """Загрузка всех обученных моделей"""
        model_files = list(self.models_dir.glob("*_physics.pt"))
        
        for model_file in model_files:
            component_name = model_file.stem.replace("model_physics", "")
            
            try:
                checkpoint = torch.load(model_file, map_location="cpu")
                # Здесь нужно загрузить правильную архитектуру модели
                # Пока просто сохраняем checkpoint
                self.models[component_name] = checkpoint
                print(f"✅ Loaded model: {component_name}")
            except Exception as e:
                print(f"⚠️ Failed to load {component_name}: {e}")
    
    def diagnose_system(
        self,
        symptom: str,
        sensor_data: Dict[str, Dict],
        equipment_config: Optional[Dict] = None
    ) -> Dict:
        """
        Главный метод диагностики системы
        
        Args:
            symptom: Описание проблемы (например, "boom_rotation_weak")
            sensor_data: {
                "pump": {"pressure_outlet": 180, "speed_rpm": 1800, ...},
                "swing_motor": {"speed_rpm": 450, "temperature": 80, ...}
            }
            equipment_config: Конфигурация оборудования (опционально)
        
        Returns:
            Полный диагностический отчёт
        """
        # 1. Запускаем ML диагностику на компонентах с датчиками
        observations = []
        
        for component, data in sensor_data.items():
            if component in self.models:
                result = self._diagnose_component(component, data, equipment_config)
                observations.append(result)
        
        # 2. Байесовская inference для всей системы
        diagnostic_results = self.bayesian_engine.diagnose(symptom, observations)
        
        # 3. Формируем полный отчёт
        report = {
            "symptom": symptom,
            "timestamp": "2025-11-08T20:00:00Z",
            "components_tested": list(sensor_data.keys()),
            "components_inferred": [
                r.component for r in diagnostic_results if r.evidence_type == "inferred"
            ],
            "diagnostics": [
                {
                    "component": r.component,
                    "fault_probability": round(r.fault_probability, 3),
                    "confidence": round(r.confidence, 3),
                    "evidence_type": r.evidence_type,
                    "severity": r.severity,
                    "reasoning": r.reasoning,
                    "recommendations": r.recommendations
                }
                for r in diagnostic_results
            ],
            "critical_findings": [
                r for r in diagnostic_results 
                if r.severity == "critical" and r.fault_probability > 0.3
            ],
            "recommended_actions": self._prioritize_actions(diagnostic_results)
        }
        
        return report
    
    def _diagnose_component(
        self,
        component: str,
        sensor_data: Dict,
        equipment_config: Optional[Dict]
    ) -> ComponentObservation:
        """ML диагностика одного компонента"""
        # TODO: Implement actual ML inference
        # Пока mock
        
        # Пример: детектируем аномалию если температура > 75
        fault_detected = sensor_data.get("temperature", 0) > 75
        confidence = 0.85 if fault_detected else 0.92
        
        return ComponentObservation(
            component=component,
            fault_detected=fault_detected,
            confidence=confidence,
            fault_type="overheating" if fault_detected else None,
            sensor_readings=sensor_data
        )
    
    def _prioritize_actions(self, results: List) -> List[str]:
        """Приоритизация действий диагноста"""
        actions = []
        
        # Критичные компоненты с высокой вероятностью
        critical = [r for r in results if r.severity == "critical" and r.fault_probability > 0.4]
        if critical:
            actions.append(
                f"🔴 КРИТИЧНО: Немедленно проверить {', '.join(r.component for r in critical[:2])}"
            )
        
        # Ненаблюдаемые компоненты с высокой вероятностью
        inferred_high = [r for r in results if r.evidence_type == "inferred" and r.fault_probability > 0.3]
        if inferred_high:
            actions.append(
                f"🟡 Установить датчики на {', '.join(r.component for r in inferred_high[:2])} для точной диагностики"
            )
        
        return actions


if __name__ == "__main__":
    # Пример использования
    system = PartialDiagnosticSystem()
    
    # Сценарий: Слабое усилие поворота, датчики на насосе и моторе
    report = system.diagnose_system(
        symptom="boom_rotation_weak",
        sensor_data={
            "pump": {
                "pressure_outlet": 180,
                "speed_rpm": 1800,
                "temperature": 65,
                "vibration": 2.1,
                "power": 45
            },
            "swing_motor": {
                "speed_rpm": 450,
                "temperature": 80,
                "pressure_inlet": 175,
                "vibration": 5.2
            }
        }
    )
    
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))
