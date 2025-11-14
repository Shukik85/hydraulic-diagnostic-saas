# services/equipment_service/openapi_config.py
"""
OpenAPI configuration для Equipment Service.
"""
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi


def custom_openapi(app: FastAPI):
    """
    Кастомная OpenAPI schema с примерами и расширениями.
    """
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Equipment Service API",
        version="1.0.0",
        description="""
# Equipment Management Service

Управление гидравлическими системами, компонентами и сенсорами.

## Features
- ✅ CRUD operations для equipment
- ✅ Hierarchical structure (System → Component → Sensor)
- ✅ Multi-tenancy support
- ✅ Real-time health monitoring
- ✅ Batch data upload

## Authentication
All endpoints require JWT Bearer token:
```
Authorization: Bearer <token>
```
        """,
        routes=app.routes,
        contact={
            "name": "Hydraulic Diagnostics Support",
            "email": "support@hydraulic-diagnostics.com",
            "url": "https://hydraulic-diagnostics.com/support"
        },
        license_info={
            "name": "Proprietary",
            "url": "https://hydraulic-diagnostics.com/license"
        },
        servers=[
            {
                "url": "https://api.hydraulic-diagnostics.com/v1/equipment",
                "description": "Production"
            },
            {
                "url": "https://staging-api.hydraulic-diagnostics.com/v1/equipment",
                "description": "Staging"
            },
            {
                "url": "http://localhost:8002",
                "description": "Local Development"
            }
        ]
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token from /auth/login endpoint"
        }
    }
    
    # Apply security globally
    openapi_schema["security"] = [{"bearerAuth": []}]
    
    # Add tags
    openapi_schema["tags"] = [
        {
            "name": "Systems",
            "description": "Hydraulic system operations"
        },
        {
            "name": "Components",
            "description": "Component management"
        },
        {
            "name": "Sensors",
            "description": "Sensor configuration and data"
        },
        {
            "name": "Health",
            "description": "System health and monitoring"
        }
    ]
    
    # Add custom extensions
    openapi_schema["x-logo"] = {
        "url": "https://hydraulic-diagnostics.com/logo.png",
        "altText": "Hydraulic Diagnostics"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def add_openapi_examples(app: FastAPI):
    """
    Добавить примеры request/response для документации.
    """
    # Equipment examples
    EQUIPMENT_EXAMPLE = {
        "equipment_id": "exc_001",
        "name": "Excavator CAT-320D #001",
        "equipment_type": "excavator",
        "model": "CAT-320D",
        "manufacturer": "Caterpillar",
        "serial_number": "CAT123456789",
        "year": 2020,
        "operating_hours": 8500,
        "components": [
            {
                "component_id": "pump_001",
                "component_type": "main_pump",
                "name": "Main Hydraulic Pump",
                "model": "Bosch Rexroth A4VSO 125",
                "sensors": [
                    {
                        "sensor_id": "sensor_pressure_001",
                        "sensor_type": "pressure",
                        "unit": "bar",
                        "min_value": 0,
                        "max_value": 350,
                        "warning_threshold": 300,
                        "critical_threshold": 320
                    }
                ]
            }
        ]
    }
    
    # Apply examples to schemas
    if hasattr(app, 'openapi_schema') and app.openapi_schema:
        schemas = app.openapi_schema.get('components', {}).get('schemas', {})
        if 'Equipment' in schemas:
            schemas['Equipment']['example'] = EQUIPMENT_EXAMPLE
    
    return app
