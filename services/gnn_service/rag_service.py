"""
RAG Service: LLM-based interpretation of GNN outputs
Интерпретация состояния компонентов на основе health_score + degradation_rate

Supports:
- OpenAI GPT-4
- Anthropic Claude
- Local Ollama (llama3, mistral, etc.)
"""

import json
import logging
import os
from typing import Dict, Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai, anthropic, ollama
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))

# Initialize LLM clients
llm_client = None

if LLM_PROVIDER == "openai" and OpenAI:
    llm_client = OpenAI(api_key=LLM_API_KEY)
elif LLM_PROVIDER == "anthropic" and Anthropic:
    llm_client = Anthropic(api_key=LLM_API_KEY)
else:
    logger.warning(f"LLM provider {LLM_PROVIDER} not configured or library not installed")


def build_rag_prompt(
    component: str,
    health_score: float,
    degradation_rate: float,
    historical_context: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build RAG prompt for LLM interpretation.
    
    Args:
        component: Component name (e.g., "pump")
        health_score: 0-1 (1 = healthy)
        degradation_rate: Derivative (negative = degrading)
        historical_context: Optional historical data
        metadata: Optional equipment metadata
    
    Returns:
        Formatted prompt string
    """
    # Historical context
    hist_str = ""
    if historical_context:
        hist_str = f"""
Historical Context:
- Trend: {historical_context.get('trend', 'unknown')}
- Operating hours: {historical_context.get('operating_hours', 'N/A')}
- Last maintenance: {historical_context.get('last_maintenance', 'N/A')}
"""
    
    # Metadata
    meta_str = ""
    if metadata:
        meta_str = f"""
Equipment Specifications:
- Type: {metadata.get('equipment_type', 'hydraulic system')}
- Operating range: {metadata.get('operating_range', 'standard')}
- Critical threshold: {metadata.get('critical_threshold', 0.3)}
"""
    
    prompt = f"""
You are an expert hydraulic equipment diagnostics system. Analyze the following component data and provide a structured assessment.

Component: {component}

Current Metrics:
- Health Score: {health_score:.3f} (0 = failed, 1 = perfectly healthy)
- Degradation Rate: {degradation_rate:.4f} per 5 minutes (negative = degrading)
{hist_str}{meta_str}

Based on this data, provide a diagnostic assessment in the following JSON format:

{{
  "state": "<one of: critical, pre_failure, degraded, warning, healthy>",
  "time_to_failure_min": <estimated minutes until failure, or null if healthy>,
  "recommended_action": "<specific maintenance action>",
  "explanation": "<brief technical explanation of your assessment>",
  "confidence": <0-1 confidence score>
}}

Classification Guidelines:
- critical: health < 0.3, immediate action required
- pre_failure: health 0.3-0.5, degrading rapidly
- degraded: health 0.5-0.7, noticeable decline
- warning: health 0.7-0.85, minor concerns
- healthy: health > 0.85, normal operation

Consider degradation_rate for urgency:
- Fast degradation (< -0.05) = more urgent
- Slow degradation (> -0.02) = can plan maintenance
- Stable/improving (>= 0) = monitor only

Provide ONLY the JSON response, no additional text.
"""
    
    return prompt


def interpret_component_state(
    component: str,
    health: float,
    degradation: float,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Interpret component state using LLM/RAG.
    
    Args:
        component: Component name
        health: Health score (0-1)
        degradation: Degradation rate (derivative)
        context: Optional historical/metadata context
    
    Returns:
        {
            "state": str,
            "time_to_failure_min": float or None,
            "recommended_action": str,
            "explanation": str,
            "confidence": float
        }
    """
    # Fallback to rule-based if LLM not available
    if llm_client is None:
        logger.warning("LLM not configured, using rule-based interpretation")
        return rule_based_interpretation(component, health, degradation)
    
    try:
        # Build prompt
        prompt = build_rag_prompt(
            component=component,
            health_score=health,
            degradation_rate=degradation,
            historical_context=context or {},
            metadata={},
        )
        
        # Call LLM
        if LLM_PROVIDER == "openai":
            response = llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert hydraulic diagnostics system."},
                    {"role": "user", "content": prompt},
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=500,
            )
            result_text = response.choices[0].message.content
        
        elif LLM_PROVIDER == "anthropic":
            response = llm_client.messages.create(
                model=LLM_MODEL,
                max_tokens=500,
                temperature=LLM_TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
            )
            result_text = response.content[0].text
        
        else:
            raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")
        
        # Parse JSON response
        result = json.loads(result_text)
        return result
        
    except Exception as e:
        logger.error(f"RAG interpretation failed: {e}")
        # Fallback to rule-based
        return rule_based_interpretation(component, health, degradation)


def rule_based_interpretation(
    component: str,
    health: float,
    degradation: float,
) -> Dict[str, Any]:
    """
    Fallback rule-based interpretation (no LLM).
    """
    # Classify state
    if health < 0.3:
        state = "critical"
        action = "Emergency stop required"
    elif health < 0.5:
        state = "pre_failure"
        action = "Schedule immediate maintenance"
    elif health < 0.7:
        state = "degraded"
        action = "Plan maintenance within 24 hours"
    elif health < 0.85:
        state = "warning"
        action = "Monitor closely, prepare spare parts"
    else:
        state = "healthy"
        action = "Continue normal operation"
    
    # Estimate time to failure (simple linear extrapolation)
    time_to_failure = None
    if degradation < 0 and health < 0.9:
        # Time to reach critical threshold (0.3)
        steps_to_critical = (health - 0.3) / abs(degradation)
        time_to_failure = steps_to_critical * 5  # 5 min per step
    
    return {
        "state": state,
        "time_to_failure_min": time_to_failure,
        "recommended_action": action,
        "explanation": f"{component.capitalize()} health: {health:.2f}, degradation: {degradation:.4f}/5min",
        "confidence": 0.7,  # Rule-based = medium confidence
    }


if __name__ == "__main__":
    # Test RAG service
    logging.basicConfig(level=logging.INFO)
    
    # Test case: degraded pump
    result = interpret_component_state(
        component="pump",
        health=0.42,
        degradation=-0.08,
        context={"trend": "declining", "operating_hours": 1250},
    )
    
    print("\n" + "="*60)
    print("RAG INTERPRETATION TEST")
    print("="*60)
    print(json.dumps(result, indent=2))
    print("="*60)
