"""FastAPI main application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging.config

from app.config import settings

logging.config.dictConfig(settings.LOGGING_CONFIG)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hydraulic Diagnostic Service",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Application shutting down...")

@app.get("/health")
async def health():
    return {"status": "alive"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}
