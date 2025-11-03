"""
ML Inference Service - FastAPI Application (Production-ready)
Enterprise гидравлическая диагностика с ML
"""

import time
import uuid
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from api.routes import router as api_router
from config import ANOMALY_THRESHOLDS, settings
from models.ensemble import EnsembleModel
from services.cache_service import CacheService
from services.health_check import HealthCheckService
from services.monitoring import setup_metrics

# Настройка логирования
logger = structlog.get_logger()

# Глобальные сервисы
ensemble_model: EnsembleModel | None = None
cache_service: CacheService | None = None
health_service: HealthCheckService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Управление жизненным циклом приложения (prod)."""
    global ensemble_model, cache_service, health_service

    logger.info("Starting ML Inference Service", version=settings.version)

    try:
        # Инициализация сервисов
        cache_service = CacheService()
        await cache_service.connect()

        health_service = HealthCheckService()

        # Загрузка ML моделей
        logger.info("Loading ML models", model_path=settings.model_path)
        ensemble_model = EnsembleModel()
        await ensemble_model.load_models()

        # Прогрев моделей
        logger.info("Warming up models")
        await ensemble_model.warmup()

        # Настройка метрик
        setup_metrics()

        logger.info("ML Service started successfully")
        yield

    except Exception as e:
        logger.error("Failed to start ML Service", error=str(e))
        raise
    finally:
        # Очистка ресурсов
        logger.info("Shutting down ML Service")
        if cache_service:
            await cache_service.disconnect()
        if ensemble_model:
            await ensemble_model.cleanup()
        logger.info("ML Service shutdown complete")


# Создание FastAPI приложения
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Enterprise ML inference service for hydraulic systems anomaly detection",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan,
)

# Сжатие ответов
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS (только проверенные источники)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# API роуты
app.include_router(api_router, prefix="/api/v1")

# Prometheus метрики
if settings.enable_metrics:
    prometheus_app = make_asgi_app()
    app.mount("/metrics", prometheus_app)


# Базовые эндпоинты
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "operational",
        "timestamp": time.time(),
    }


@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса."""
    if not health_service:
        raise HTTPException(status_code=503, detail="Health service not initialized")

    health_status = await health_service.check_health()
    return health_status


@app.get("/ready")
async def readiness_check():
    """Проверка готовности к обработке запросов."""
    if not ensemble_model or not ensemble_model.is_ready():
        raise HTTPException(status_code=503, detail="Models not ready")

    if not cache_service or not await cache_service.is_connected():
        raise HTTPException(status_code=503, detail="Cache service not ready")

    return {
        "ready": True,
        "models_loaded": ensemble_model.get_loaded_models(),
        "cache_connected": await cache_service.is_connected(),
        "timestamp": time.time(),
    }


@app.get("/info")
async def service_info():
    """Информация о сервисе."""
    model_info = ensemble_model.get_model_info() if ensemble_model else {}

    return {
        "service": settings.app_name,
        "version": settings.version,
        "debug": settings.debug,
        "models": model_info,
        "thresholds": ANOMALY_THRESHOLDS,
        "performance": {
            "max_inference_time_ms": settings.max_inference_time_ms,
            "batch_size": settings.batch_size,
            "cache_enabled": settings.cache_predictions,
            "cache_ttl": settings.cache_ttl_seconds,
        },
        "timestamp": time.time(),
    }


# Глобальные dependency функции

def get_ensemble_model() -> EnsembleModel:
    if not ensemble_model:
        raise HTTPException(status_code=503, detail="Ensemble model not loaded")
    return ensemble_model


def get_cache_service() -> CacheService:
    if not cache_service:
        raise HTTPException(status_code=503, detail="Cache service not initialized")
    return cache_service


# Глобальный обработчик ошибок (структурированный ответ)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    trace_id = str(uuid.uuid4())
    logger.error(
        "Unhandled exception",
        error=str(exc),
        error_type=type(exc).__name__,
        trace_id=trace_id,
        url=str(request.url),
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "trace_id": trace_id,
            "timestamp": time.time(),
        },
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=1 if settings.debug else settings.workers,
        log_level=settings.log_level.lower(),
        reload=settings.debug,
        access_log=settings.debug,
    )
