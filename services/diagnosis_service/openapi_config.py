# services/diagnosis_service/openapi_config.py
"""
OpenAPI configuration для Diagnosis Service.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI):
    """
    Custom OpenAPI schema для Diagnosis Service.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Diagnosis Service API",
        version="1.0.0",
        description="""
# Hydraulic Diagnosis Orchestration Service

Оркестрирует полный пайплайн диагностики:
1. Сбор sensor data из TimescaleDB
2. GNN inference
3. RAG interpretation
4. Результаты + рекомендации

## Features
- ✅ Full diagnosis pipeline
- ✅ Batch processing
- ✅ Real-time progress tracking
- ✅ Historical data analysis
- ✅ WebSocket updates

## Authentication
Requires JWT Bearer token.
        """,
        routes=app.routes,
        servers=[
            {"url": "https://api.hydraulic-diagnostics.com/v1/diagnosis", "description": "Production"},
            {"url": "http://localhost:8003", "description": "Development"}
        ]
    )
    
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {"type": "http", "scheme": "bearer", "bearerFormat": "JWT"}
    }
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    openapi_schema["tags"] = [
        {"name": "Diagnosis", "description": "Diagnosis operations"},
        {"name": "Monitoring", "description": "Health and metrics"}
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
