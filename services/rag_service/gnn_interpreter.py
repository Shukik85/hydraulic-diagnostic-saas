# services/rag_service/gnn_interpreter.py
"""
Интерпретация результатов GNN через DeepSeek-R1 reasoning.
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

from model_loader import get_model

logger = logging.getLogger(__name__)


class GNNInterpreter:
    """
    Интерпретация GNN outputs с reasoning объяснениями.
    """
    
    SYSTEM_PROMPT = """
Ты эксперт по диагностике гидравлических систем с глубокими знаниями в:
- Гидравлике мобильных машин (экскаваторы, погрузчики)
- Режимах отказа гидравлических компонентов
- Анализе временных рядов sensor data
- Predictive maintenance стратегиях

Твоя задача: интерпретировать результаты ML модели (GNN) и давать:
1. Понятное объяснение состояния системы
2. Пошаговый reasoning анализа проблем
3. Приоритизированные рекомендации
4. Прогноз развития ситуации

Всегда используй тег <думает>...</думает> для reasoning процесса.
"""
    
    def __init__(self):
        self.model = get_model()
        logger.info("GNNInterpreter initialized")
    
    def interpret_diagnosis(
        self,
        gnn_result: Dict,
        equipment_context: Dict,
        historical_context: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Полная интерпретация GNN diagnosis.
        
        Args:
            gnn_result: Output from GNN Service
            equipment_context: Equipment metadata and specs
            historical_context: Previous diagnoses (optional)
            
        Returns:
            dict: Comprehensive interpretation with reasoning
        """
        logger.info(f"Interpreting diagnosis for {gnn_result.get('equipment_id')}")
        
        # Формируем prompt
        prompt = self._build_diagnosis_prompt(
            gnn_result,
            equipment_context,
            historical_context
        )
        
        # Generate response с reasoning
        response = self.model.generate(
            prompt,
            max_tokens=2048,
            temperature=0.7
        )
        
        # Parse response
        interpretation = self._parse_response(response)
        
        # Add metadata
        interpretation["timestamp"] = datetime.utcnow().isoformat()
        interpretation["model"] = "DeepSeek-R1-Distill-32B"
        interpretation["gnn_request_id"] = gnn_result.get("request_id")
        
        logger.info("Interpretation completed")
        return interpretation
    
    def _build_diagnosis_prompt(self, gnn_result, context, history):
        """
        Строим comprehensive prompt для модели.
        """
        prompt = f"{self.SYSTEM_PROMPT}\n\n"
        
        # Equipment info
        prompt += "=== ИНФОРМАЦИЯ ОБ ОБОРУДОВАНИИ ===\n"
        prompt += f"ID: {context.get('equipment_id')}\n"
        prompt += f"Тип: {context.get('equipment_type')}\n"
        prompt += f"Модель: {context.get('model')}\n"
        prompt += f"Производитель: {context.get('manufacturer')}\n"
        prompt += f"Год выпуска: {context.get('year')}\n"
        prompt += f"Наработка: {context.get('operating_hours', 0)} ч\n\n"
        
        # GNN Results
        prompt += "=== РЕЗУЛЬТАТЫ GNN ДИАГНОСТИКИ ===\n"
        prompt += f"Общее состояние: {gnn_result.get('overall_health_score', 0)*100:.1f}%\n\n"
        
        # Component health
        if "component_health" in gnn_result:
            prompt += "Состояние компонентов:\n"
            for comp in gnn_result["component_health"]:
                health = comp.get("health_score", 0) * 100
                deg_rate = comp.get("degradation_rate", 0)
                prompt += f"  • {comp.get('component_type')}: {health:.1f}% "
                prompt += f"(деградация: {deg_rate:.3f}/день)\n"
                if comp.get("anomalies"):
                    for anomaly in comp["anomalies"]:
                        prompt += f"    - {anomaly}\n"
            prompt += "\n"
        
        # Anomalies
        if gnn_result.get("anomalies"):
            prompt += "Обнаруженные аномалии:\n"
            for anomaly in gnn_result["anomalies"]:
                prompt += f"  • Тип: {anomaly.get('anomaly_type')}\n"
                prompt += f"    Severity: {anomaly.get('severity')}\n"
                prompt += f"    Уверенность: {anomaly.get('confidence', 0)*100:.0f}%\n"
                if anomaly.get('description'):
                    prompt += f"    Описание: {anomaly['description']}\n"
                if anomaly.get('affected_components'):
                    prompt += f"    Затронутые компоненты: {', '.join(anomaly['affected_components'])}\n"
            prompt += "\n"
        
        # Historical context
        if history and len(history) > 0:
            prompt += "=== ИСТОРИЯ ДИАГНОСТИК ===\n"
            prompt += f"Предыдущих диагностик: {len(history)}\n"
            
            # Last diagnosis
            last = history[-1]
            prompt += f"Последняя: {last.get('timestamp')}\n"
            prompt += f"  Состояние было: {last.get('overall_health', 0)*100:.1f}%\n"
            if last.get('issues'):
                prompt += f"  Проблемы: {', '.join(last['issues'])}\n"
            prompt += "\n"
        
        # Task
        prompt += "=== ЗАДАЧА ===\n"
        prompt += "Проанализируй данные и предоставь:\n"
        prompt += "1. ПОНЯТНОЕ РЕЗЮМЕ состояния (для оператора)\n"
        prompt += "2. ПОШАГОВЫЙ АНАЛИЗ проблем (используй <думает>)\n"
        prompt += "3. ПРИОРИТИЗИРОВАННЫЕ РЕКОМЕНДАЦИИ\n"
        prompt += "4. ПРОГНОЗ развития (когда ожидать отказа)\n\n"
        
        prompt += "Ответ давай НА РУССКОМ ЯЗЫКЕ.\n"
        prompt += "Используй технические термины, но объясняй понятно.\n\n"
        
        return prompt
    
    def _parse_response(self, response: str) -> Dict:
        """
        Парсим structured ответ из text.
        """
        # Extract reasoning
        reasoning = ""
        if "<думает>" in response and "</думает>" in response:
            start = response.index("<думает>") + len("<думает>")
            end = response.index("</думает>")
            reasoning = response[start:end].strip()
        
        # Extract sections
        sections = {
            "summary": "",
            "reasoning": reasoning,
            "analysis": "",
            "recommendations": [],
            "prognosis": ""
        }
        
        # Simple parsing (можно улучшить с regex)
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith("<думает") or line.startswith("</думает"):
                continue
            
            # Detect sections
            if "РЕЗЮМЕ" in line.upper() or "SUMMARY" in line.upper():
                current_section = "summary"
            elif "АНАЛИЗ" in line.upper() or "ANALYSIS" in line.upper():
                current_section = "analysis"
            elif "РЕКОМЕНДАЦИ" in line.upper() or "RECOMMENDATION" in line.upper():
                current_section = "recommendations"
            elif "ПРОГНОЗ" in line.upper() or "PROGNOSIS" in line.upper():
                current_section = "prognosis"
            elif current_section:
                if current_section == "recommendations":
                    if line.startswith(("-", "•", "*", "1.", "2.", "3.")):
                        # Extract recommendation
                        rec = line.lstrip("-•*123456789. ")
                        if rec:
                            sections["recommendations"].append(rec)
                else:
                    sections[current_section] += line + "\n"
        
        # Clean up
        for key in ["summary", "analysis", "prognosis"]:
            sections[key] = sections[key].strip()
        
        return sections
    
    def explain_anomaly(
        self,
        anomaly_type: str,
        context: Dict
    ) -> str:
        """
        Детальное объяснение конкретной аномалии.
        
        Args:
            anomaly_type: Type of anomaly (e.g., 'pressure_drop')
            context: Additional context
            
        Returns:
            str: Detailed explanation
        """
        prompt = f"""{self.SYSTEM_PROMPT}

Объясни детально аномалию типа '{anomaly_type}' в гидравлической системе.

Контекст:
{json.dumps(context, indent=2, ensure_ascii=False)}

Дай:
1. Что это значит
2. Возможные причины (3-5)
3. Как диагностировать
4. Срочность проблемы
5. Что будет, если не исправить

Используй <думает> для reasoning.
Ответ на РУССКОМ.
"""
        
        response = self.model.generate(prompt, max_tokens=1024)
        return response
    
    def compare_diagnoses(
        self,
        current: Dict,
        previous: Dict
    ) -> str:
        """
        Сравнение текущей и предыдущей диагностики.
        
        Args:
            current: Current diagnosis
            previous: Previous diagnosis
            
        Returns:
            str: Comparison analysis
        """
        prompt = f"""{self.SYSTEM_PROMPT}

Сравни две диагностики одной системы и объясни изменения.

Предыдущая диагностика:
{json.dumps(previous, indent=2, ensure_ascii=False)}

Текущая диагностика:
{json.dumps(current, indent=2, ensure_ascii=False)}

Проанализируй:
1. Что изменилось (улучшение/ухудшение)
2. Скорость деградации
3. Эффективность принятых мер (если были)
4. Новые риски
5. Обновлённые рекомендации

Используй <думает> для reasoning.
Ответ на РУССКОМ.
"""
        
        response = self.model.generate(prompt, max_tokens=1536)
        return response


# Global interpreter
_interpreter: Optional[GNNInterpreter] = None


def get_interpreter() -> GNNInterpreter:
    """
    Get global interpreter instance.
    
    Returns:
        GNNInterpreter: Initialized interpreter
    """
    global _interpreter
    if _interpreter is None:
        _interpreter = GNNInterpreter()
    return _interpreter
