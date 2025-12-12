# services/gnn_service/openapi_config.py
"""
OpenAPI configuration для GNN Service.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI):
    """
    Custom OpenAPI schema для GNN Service.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="GNN Service API",
        version="1.0.0",
        description="""
# Graph Neural Network Inference Service

ML-powered hydraulic diagnostics используя Universal Temporal GNN.

## Features
- ✅ Real-time inference (< 500ms)
- ✅ Batch processing
- ✅ Model management
- ✅ Training pipeline
- ✅ GPU acceleration

## Model
- **Architecture**: GAT + LSTM
- **Framework**: ONNX Runtime
- **GPU**: CUDA 12.1
- **Input**: Time-series sensor data + graph topology

## Authentication
Requires JWT Bearer token:
```
Authorization: Bearer <token>
```
        """,
        routes=app.routes,
        servers=[
            {"url": "https://api.hydraulic-diagnostics.com/v1/gnn", "description": "Production"},
            {"url": "http://localhost:8002", "description": "Development"}
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
        {"name": "Inference", "description": "ML inference operations"},
        {"name": "Admin", "description": "Model deployment and training (admin-only)"},
        {"name": "Monitoring", "description": "Health and metrics"}
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema
