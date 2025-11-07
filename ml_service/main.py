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

from api import routes  # for setting global adaptive_thresholds
from api.routes import router as api_router
from models.ensemble import EnsembleModel
from services.adaptive_threshold_service import AdaptiveThresholdService
from services.cache_service import CacheService
from services.health_check import HealthCheckService
from services.monitoring import setup_metrics
from src.config import ANOMALY_THRESHOLDS, settings

# Настройка логирования
logger = structlog.get_logger()

# Глобальные сервисы
ensemble_model: EnsembleModel | None = None
cache_service: CacheService | None = None
health_service: HealthCheckService | None = None
adaptive_thresholds: AdaptiveThresholdService | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Управление жизненным циклом приложения (prod)."""
    global ensemble_model, cache_service, health_service, adaptive_thresholds

    logger.info("Starting ML Inference Service", version=settings.version)

    try:
        # Инициализация сервисов
        cache_service = CacheService()
        await cache_service.connect()

        health_service = HealthCheckService()

        # Initialize adaptive thresholds
        adaptive_thresholds = AdaptiveThresholdService(cache_service=cache_service)

        # Set global reference in routes module
        routes.adaptive_thresholds = adaptive_thresholds

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
    """Проверка здоровья сервиса (resources + dependencies)."""
    if not health_service:
        raise HTTPException(status_code=503, detail="Health service not initialized")

    status = await health_service.check_health(ensemble_model, cache_service)
    return status


@app.get("/ready")
async def readiness_check():
    """Проверка готовности к обработке запросов."""
    if not ensemble_model or not ensemble_model.is_ready():
        return JSONResponse(
            content={
                "ready": False,
                "reason": "Models not loaded",
                "models_loaded": ensemble_model.get_loaded_models() if ensemble_model else [],
            },
            status_code=503,
        )

    if not cache_service or not await cache_service.is_connected():
        return JSONResponse(
            content={
                "ready": False,
                "reason": "Cache service not ready",
            },
            status_code=503,
        )

    ready_data = {
        "ready": True,
        "models_loaded": ensemble_model.get_loaded_models(),
        "ensemble_ready": ensemble_model.is_ready(),
        "cache_ready": await cache_service.is_connected(),
        "adaptive_thresholds_ready": adaptive_thresholds is not None,
        "uptime_seconds": time.time() - (getattr(app.state, "start_time", time.time())),
    }

    return JSONResponse(content=ready_data, status_code=200)


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
        "adaptive_thresholds": {
            "enabled": getattr(settings, "adaptive_thresholds_enabled", True),
            "adaptation_rate": getattr(settings, "threshold_adaptation_rate", 0.05),
            "target_fpr": getattr(settings, "target_fpr", 0.10),
        },
        "timestamp": time.time(),
    }


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
