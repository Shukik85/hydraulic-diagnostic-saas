# services/rag_service/openapi_config.py
"""
OpenAPI configuration для RAG Service.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI):
    """
    Кастомная OpenAPI schema для RAG Service.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="RAG Service API",
        version="1.0.0",
        description="""
# RAG Interpretation Service

AI-powered interpretation GNN диагностических результатов используя DeepSeek-R1.

## Features
- ✅ Reasoning-based interpretation
- ✅ Human-readable explanations
- ✅ Prioritized recommendations
- ✅ Failure prognosis
- ✅ Context-aware analysis

## Model
- **DeepSeek-R1-Distill-32B**
- **Latency**: ~2-3 seconds
- **Context**: 8K tokens
- **GPU**: 2x A100

## Authentication
Requires JWT token:
```
Authorization: Bearer <token>
```
        """,
        routes=app.routes,
        servers=[
            {"url": "https://api.hydraulic-diagnostics.com/v1/rag", "description": "Production"},
            {"url": "http://localhost:8004", "description": "Development"}
        ]
    )
    
    # Security
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Tags
    openapi_schema["tags"] = [
        {"name": "Interpretation", "description": "GNN result interpretation"},
        {"name": "Explanation", "description": "Anomaly explanations"},
        {"name": "Generation", "description": "Generic text generation"},
        {"name": "Health", "description": "Service health checks"}
    ]
    
    # Examples
    openapi_schema["components"]["examples"] = {
        "GNNResult": {
            "value": {
                "overall_health_score": 0.65,
                "component_health": [
                    {
                        "component_id": "pump_001",
                        "component_type": "main_pump",
                        "health_score": 0.65,
                        "degradation_rate": 0.08
                    }
                ],
                "anomalies": [
                    {
                        "anomaly_type": "pressure_drop",
                        "severity": "high",
                        "confidence": 0.85,
                        "affected_components": ["pump_001"]
                    }
                ]
            }
        },
        "RAGInterpretation": {
            "value": {
                "summary": "🔴 Критическое падение давления в главном насосе. Текущее состояние: 65% от номинального.",
                "reasoning": "<думает>\nШаг 1: Анализирую health_score (65%) - находится в warning zone...\nШаг 2: Обнаружена аномалия pressure_drop с высокой уверенностью (85%)...\n</думает>",
                "recommendations": [
                    "Срочная замена масляного фильтра HF-100",
                    "Проверка качества гидравлического масла",
                    "Осмотр насоса на предмет износа"
                ],
                "prognosis": "Без вмешательства ожидается полный отказ через 8-10 дней",
                "timestamp": "2025-11-13T03:00:00Z",
                "model": "DeepSeek-R1-Distill-32B"
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
