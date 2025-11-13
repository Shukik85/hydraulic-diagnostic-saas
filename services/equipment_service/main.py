# services/equipment_service/main.py
"""
Equipment Service - CRUD operations for hydraulic systems.
"""
import os
import logging
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from openapi_config import custom_openapi
from monitoring_endpoints import router as monitoring_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Equipment Service API",
    version="1.0.0",
    description="Equipment management and hierarchy service",
    openapi_version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include monitoring router
app.include_router(monitoring_router)

# Apply custom OpenAPI
app.openapi = lambda: custom_openapi(app)


# === Models ===

class Sensor(BaseModel):
    """Sensor definition."""
    sensor_id: str
    sensor_type: str
    unit: str
    min_value: float
    max_value: float
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


class Component(BaseModel):
    """Component with sensors."""
    component_id: str
    component_type: str
    name: str
    model: Optional[str] = None
    sensors: List[Sensor] = []


class Equipment(BaseModel):
    """Equipment system."""
    equipment_id: str
    name: str
    equipment_type: str
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    serial_number: Optional[str] = None
    year: Optional[int] = None
    operating_hours: Optional[float] = None
    components: List[Component] = []


class EquipmentCreate(BaseModel):
    """Create equipment request."""
    name: str
    equipment_type: str
    model: Optional[str] = None
    manufacturer: Optional[str] = None


class EquipmentUpdate(BaseModel):
    """Update equipment request."""
    name: Optional[str] = None
    operating_hours: Optional[float] = None


# === CRUD Endpoints ===

@app.post("/systems", response_model=Equipment, tags=["Systems"])
async def create_system(equipment: EquipmentCreate):
    """
    Create new hydraulic system.
    
    **Returns**: Created system with generated ID
    """
    try:
        # TODO: Save to database
        equipment_id = f"eq_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        logger.info(f"Creating system: {equipment.name}")
        
        return Equipment(
            equipment_id=equipment_id,
            name=equipment.name,
            equipment_type=equipment.equipment_type,
            model=equipment.model,
            manufacturer=equipment.manufacturer,
            components=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/systems/{equipment_id}", response_model=Equipment, tags=["Systems"])
async def get_system(equipment_id: str):
    """
    Get system by ID with full hierarchy.
    
    **Returns**: System → Components → Sensors
    """
    try:
        # TODO: Fetch from database
        logger.info(f"Fetching system: {equipment_id}")
        
        # Mock response
        return Equipment(
            equipment_id=equipment_id,
            name="Excavator CAT-320D #001",
            equipment_type="excavator",
            model="CAT-320D",
            components=[]
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail="System not found")


@app.get("/systems", response_model=List[Equipment], tags=["Systems"])
async def list_systems(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    equipment_type: Optional[str] = None
):
    """
    List all systems with pagination.
    
    **Filters**: equipment_type, manufacturer, etc.
    """
    try:
        # TODO: Query database
        logger.info(f"Listing systems (skip={skip}, limit={limit})")
        
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.patch("/systems/{equipment_id}", response_model=Equipment, tags=["Systems"])
async def update_system(equipment_id: str, update: EquipmentUpdate):
    """
    Update system properties.
    """
    try:
        logger.info(f"Updating system: {equipment_id}")
        # TODO: Update in database
        
        return await get_system(equipment_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail="System not found")


@app.delete("/systems/{equipment_id}", tags=["Systems"])
async def delete_system(equipment_id: str):
    """
    Delete system and all related data.
    
    **Warning**: This action is irreversible.
    """
    try:
        logger.info(f"Deleting system: {equipment_id}")
        # TODO: Delete from database
        
        return {"status": "deleted", "equipment_id": equipment_id}
    except Exception as e:
        raise HTTPException(status_code=404, detail="System not found")


@app.get("/systems/{equipment_id}/health", tags=["Health"])
async def get_system_health(equipment_id: str):
    """
    Get current health status of system.
    
    **Returns**: Latest health scores and component statuses.
    """
    try:
        # TODO: Query latest diagnosis results
        return {
            "equipment_id": equipment_id,
            "overall_health_score": 0.85,
            "status": "healthy",
            "last_updated": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=404, detail="System not found")


@app.post("/components/batch", tags=["Components"])
async def batch_create_components(components: List[Component]):
    """
    Batch create components.
    
    **Use case**: CSV upload, bulk import.
    """
    try:
        logger.info(f"Batch creating {len(components)} components")
        # TODO: Batch insert to database
        
        return {
            "created": len(components),
            "component_ids": [c.component_id for c in components]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sensors/validate-hierarchy", tags=["Sensors"])
async def validate_hierarchy(data: Dict):
    """
    Validate CSV hierarchical data.
    
    **Use case**: Frontend CSV upload validation.
    """
    try:
        system_id = data.get("system_id")
        components = data.get("components", [])
        sensors = data.get("sensors", [])
        
        # TODO: Validate structure
        
        return {
            "valid": True,
            "system_id": system_id,
            "components_count": len(components),
            "sensors_count": len(sensors)
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [str(e)]
        }


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "service": "Equipment Service",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
