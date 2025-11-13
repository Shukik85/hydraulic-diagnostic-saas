# services/gnn_service/main.py
"""
GNN Service - ML inference and model management.
Updated with monitoring and admin endpoints.
"""
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from monitoring_endpoints import router as monitoring_router
from admin_endpoints import router as admin_router
from openapi_config import custom_openapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="GNN Service API",
    version="1.0.0",
    description="Graph Neural Network inference and training service",
    openapi_version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(monitoring_router)
app.include_router(admin_router)

# Apply custom OpenAPI
app.openapi = lambda: custom_openapi(app)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "GNN Service",
        "version": "1.0.0",
        "status": "operational"
    }


# === Existing inference endpoints ===
# (Keep existing /inference, /batch-inference, etc.)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8002"))
    uvicorn.run(app, host="0.0.0.0", port=port)
