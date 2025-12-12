"""FastAPI main application (DEPRECATED).

WARNING: This is legacy code from old API (port 8002).

Replaced by app/main.py with modern implementation.
Preserved for historical reference only.

Production-ready inference API:
- Async endpoints
- Error handling
- CORS
- Logging
- Health checks (basic + detailed)
- Model versioning
- Request ID tracking

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

# NOTE: Full implementation archived. See app/main.py for current API.
# This file preserved in _deprecated/ for historical reference.
# Do NOT use in production.

from __future__ import annotations

import logging
import sys
import time
import psutil
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from api.middleware import RequestIDMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global state
engine: object | None = None
topology: object | None = None
model_manager: object | None = None
start_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events (DEPRECATED)."""
    # Startup
    global engine, topology, model_manager, start_time
    
    start_time = time.time()
    
    logger.info("Initializing GNN Inference Service...", extra={"request_id": "startup"})
    logger.warning("This API is DEPRECATED. Use app/main.py instead!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down GNN Inference Service...", extra={"request_id": "shutdown"})


# Create FastAPI app
app = FastAPI(
    title="GNN Inference Service (DEPRECATED)",
    description="Legacy API - DO NOT USE IN PRODUCTION",
    version="1.0.0-deprecated",
    lifespan=lifespan
)

# Add middlewares
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "deprecated",
        "message": "This API is deprecated. Use new API at port 8000",
        "deprecated_version": "1.0.0",
        "new_api_url": "http://localhost:8000/docs"
    }


if __name__ == "__main__":
    logger.error("DEPRECATED API - DO NOT RUN IN PRODUCTION")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="warning"
    )
