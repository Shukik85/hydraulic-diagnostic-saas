"""
FastAPI Backend Core - Production-ready microservice
Enterprise-grade hydraulic diagnostics platform
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import structlog

from .config import settings
from .api import metadata, ingestion, health, users
from .db.session import engine, Base
from .middleware.auth import AuthMiddleware
from .middleware.quota import QuotaMiddleware

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management: startup and shutdown events"""
    # Startup
    logger.info("backend_fastapi_starting", version=settings.VERSION)

    # Create tables (for development; use Alembic in production)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info("backend_fastapi_ready")

    yield

    # Shutdown
    logger.info("backend_fastapi_shutting_down")
    await engine.dispose()


app = FastAPI(
    title="Hydraulic Diagnostics Backend",
    version=settings.VERSION,
    description="Core API for equipment metadata, sensor ingestion, and ML orchestration",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(AuthMiddleware)
app.add_middleware(QuotaMiddleware)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", exc_info=exc, path=request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(metadata.router, prefix="/metadata", tags=["Metadata"])
app.include_router(ingestion.router, prefix="/ingestion", tags=["Ingestion"])


@app.get("/")
async def root():
    return {
        "service": "backend_fastapi",
        "version": settings.VERSION,
        "status": "operational"
    }
