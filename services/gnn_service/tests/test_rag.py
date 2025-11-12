"""
Test suite for RAG service
"""
import pytest
from rag_service import (
    interpret_component_state,
    rule_based_interpretation,
    build_rag_prompt,
)

def test_rule_based_interpretation():
    """Test fallback rule-based interpretation."""
    
    # Healthy component
    result = rule_based_interpretation("pump", health=0.95, degradation=-0.01)
    assert result["state"] == "healthy"
    assert result["time_to_failure_min"] is None
    
    # Critical component
    result = rule_based_interpretation("valve", health=0.25, degradation=-0.05)
    assert result["state"] == "critical"
    
    # Degraded component
    result = rule_based_interpretation("motor", health=0.55, degradation=-0.08)
    assert result["state"] == "degraded"
    assert result["time_to_failure_min"] is not None

def test_rag_prompt_building():
    """Test RAG prompt construction."""
    prompt = build_rag_prompt(
        component="pump",
        health_score=0.42,
        degradation_rate=-0.08,
        historical_context={"trend": "declining"},
        metadata={"equipment_type": "excavator"},
    )
    
    assert "pump" in prompt.lower()
    assert "0.42" in prompt or "0.420" in prompt
    assert "declining" in prompt.lower()
    assert "excavator" in prompt.lower()

def test_interpret_component_state():
    """Test component state interpretation (uses fallback if LLM not configured)."""
    result = interpret_component_state(
        component="pump",
        health=0.65,
        degradation=-0.05,
        context={},
    )
    
    assert "state" in result
    assert result["state"] in ["critical", "pre_failure", "degraded", "warning", "healthy"]
    assert "recommended_action" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
