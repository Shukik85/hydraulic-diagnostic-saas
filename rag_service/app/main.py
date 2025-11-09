"""RAG Service with DeepSeek-R1 - Main FastAPI application."""

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import health, rag

# Setup logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger()

# Create FastAPI app
app = FastAPI(
    title="RAG Service (DeepSeek-R1)",
    description="Retrieval-Augmented Generation for Hydraulic Diagnostics",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])

@app.on_event("startup")
async def startup():
    logger.info("RAG Service starting", model=settings.llm_model)

@app.on_event("shutdown")
async def shutdown():
    logger.info("RAG Service shutting down")
