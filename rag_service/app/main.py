"""FastAPI main application for RAG internal microservice."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .config import get_settings
from .routes import rag, health

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("RAG Service starting up...")
    # Load models, initialize vector stores, etc.
    yield
    logger.info("RAG Service shutting down...")


app = FastAPI(
    title="RAG Internal Service",
    description="Internal RAG microservice for Hydraulic Diagnostic Platform",
    version="0.1.0",
    docs_url=None,  # Disable Swagger UI (internal only)
    redoc_url=None,  # Disable ReDoc (internal only)
    lifespan=lifespan,
)

# CORS (только для internal network)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://backend:8000"],  # Только backend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "RAG Internal Service",
        "version": "0.1.0",
        "status": "running",
    }
