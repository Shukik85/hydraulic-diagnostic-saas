"""
Hydraulic System Knowledge Graph
Описывает взаимосвязи компонентов и типичные неисправности
"""
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class FaultCause:
    component: str
    symptom: str
    prior_probability: float
    severity: str  # "critical", "warning", "info"
    description: str


@dataclass
class Symptom:
    name: str
    description: str
    possible_causes: List[FaultCause]
    required_sensors: List[str]
    optional_sensors: List[str]


class HydraulicSystemKnowledgeGraph:
    """Экспертные знания о гидравлической системе"""
    
    def __init__(self):
        self.symptoms = self._build_knowledge_base()
    
    def _build_knowledge_base(self) -> Dict[str, Symptom]:
        return {
            "boom_rotation_weak": Symptom(
                name="boom_rotation_weak",
                description="Недостаточное усилие поворота стрелы",
                required_sensors=["pump", "swing_motor"],
                optional_sensors=["valve", "filter", "boom_cylinder"],
                possible_causes=[
                    FaultCause(
                        component="pump",
                        symptom="low_pressure",
                        prior_probability=0.35,
                        severity="critical",
                        description="Насос не развивает номинальное давление"
                    ),
                    FaultCause(
                        component="swing_motor",
                        symptom="high_friction",
                        prior_probability=0.30,
                        severity="critical",
                        description="Повышенное трение в гидромоторе поворота"
                    ),
                    FaultCause(
                        component="valve",
                        symptom="restricted_flow",
                        prior_probability=0.15,
                        severity="warning",
                        description="Частичное заедание распределителя"
                    ),
                    FaultCause(
                        component="filter",
                        symptom="clogged",
                        prior_probability=0.10,
                        severity="warning",
                        description="Засорение фильтра"
                    ),
                    FaultCause(
                        component="hydraulic_line",
                        symptom="leak",
                        prior_probability=0.05,
                        severity="warning",
                        description="Утечка в гидролинии"
                    ),
                    FaultCause(
                        component="boom_cylinder",
                        symptom="leak",
                        prior_probability=0.05,
                        severity="info",
                        description="Утечка в цилиндре стрелы (косвенное влияние)"
                    )
                ]
            ),
            
            "slow_cylinder_extension": Symptom(
                name="slow_cylinder_extension",
                description="Медленное выдвижение цилиндра",
                required_sensors=["cylinder"],
                optional_sensors=["pump", "valve"],
                possible_causes=[
                    FaultCause(
                        component="cylinder",
                        symptom="internal_leak",
                        prior_probability=0.40,
                        severity="critical",
                        description="Внутренняя утечка в цилиндре"
                    ),
                    FaultCause(
                        component="pump",
                        symptom="low_flow",
                        prior_probability=0.30,
                        severity="critical",
                        description="Низкая производительность насоса"
                    ),
                    FaultCause(
                        component="valve",
                        symptom="partial_blockage",
                        prior_probability=0.20,
                        severity="warning",
                        description="Частичная блокировка клапана"
                    ),
                    FaultCause(
                        component="filter",
                        symptom="clogged",
                        prior_probability=0.10,
                        severity="warning",
                        description="Засорение фильтра"
                    )
                ]
            ),
            
            "pump_overheating": Symptom(
                name="pump_overheating",
                description="Перегрев насоса",
                required_sensors=["pump"],
                optional_sensors=["cooler", "filter"],
                possible_causes=[
                    FaultCause(
                        component="pump",
                        symptom="bearing_wear",
                        prior_probability=0.40,
                        severity="critical",
                        description="Износ подшипников насоса"
                    ),
                    FaultCause(
                        component="cooler",
                        symptom="insufficient_cooling",
                        prior_probability=0.30,
                        severity="critical",
                        description="Недостаточное охлаждение"
                    ),
                    FaultCause(
                        component="filter",
                        symptom="clogged",
                        prior_probability=0.20,
                        severity="warning",
                        description="Засорение фильтра - повышенная нагрузка"
                    ),
                    FaultCause(
                        component="hydraulic_oil",
                        symptom="degraded",
                        prior_probability=0.10,
                        severity="warning",
                        description="Деградация масла"
                    )
                ]
            )
        }
    
    def get_symptom(self, symptom_name: str) -> Optional[Symptom]:
        return self.symptoms.get(symptom_name)
    
    def get_all_symptoms(self) -> List[str]:
        return list(self.symptoms.keys())
    
    def save_to_json(self, filepath: str):
        """Сохранить knowledge graph в JSON"""
        data = {
            name: {
                "description": symptom.description,
                "required_sensors": symptom.required_sensors,
                "optional_sensors": symptom.optional_sensors,
                "possible_causes": [asdict(cause) for cause in symptom.possible_causes]
            }
            for name, symptom in self.symptoms.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    kg = HydraulicSystemKnowledgeGraph()
    kg.save_to_json("knowledge_graph.json")
    print("✅ Knowledge graph saved!")
