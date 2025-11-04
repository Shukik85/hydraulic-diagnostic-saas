from __future__ import annotations

import asyncio
import time
import uuid
from typing import Union

import numpy as np
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Response

from config import settings
from models.ensemble import EnsembleModel
from services.cache_service import CacheService
from services.feature_engineering import FeatureEngineer
from services.monitoring import metrics
from services.utils.features import vector_from_feature_vector, build_feature_cache_key

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ConfigResponse,
    ConfigUpdateRequest,
    ErrorResponse,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    MetricsResponse,
    ModelStatusResponse,
    PredictionRequest,
    PredictionResponse,
    ModelPrediction,
    AnomalyPrediction,
)

logger = structlog.get_logger()
router = APIRouter()


# Dependency injection
def get_ensemble_model() -> EnsembleModel:
    from main import ensemble_model

    if not ensemble_model or not ensemble_model.is_ready():
        raise HTTPException(status_code=503, detail="Ensemble model not ready")
    return ensemble_model


def get_cache_service() -> CacheService:
    from main import cache_service

    if not cache_service:
        raise HTTPException(status_code=503, detail="Cache service not available")
    return cache_service


async def _predict_core(req: PredictionRequest, ensemble: EnsembleModel, cache: CacheService) -> PredictionResponse:
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    # Извлечение признаков
    feature_engineer = FeatureEngineer()
    fv = await feature_engineer.extract_features(req.sensor_data)
    vector = vector_from_feature_vector(fv)

    # Кеш
    cache_key = None
    if req.use_cache and settings.cache_predictions:
        cache_key = build_feature_cache_key(vector, fv.feature_names)
        cached = await cache.get_prediction(cache_key)
        if cached:
            logger.info("Cache hit", trace_id=trace_id, cache_key=cache_key)
            metrics.cache_hits.inc()
            cached["trace_id"] = trace_id
            cached["timestamp"] = time.time()
            cached["cache_hit"] = True
            return PredictionResponse(**cached)

    # Предсказание
    vector = np.asarray(vector, dtype=float)
    result = await ensemble.predict(vector)

    # Сборка ответа согласно схемам
    prediction = AnomalyPrediction(
        is_anomaly=bool(result.get("is_anomaly", False)),
        anomaly_score=float(result.get("ensemble_score", 0.5)),
        severity=str(result.get("severity", "normal")),
        confidence=float(result.get("confidence", 0.8)),
        affected_components=[],
        anomaly_type=None,
    )

    ml_predictions: list[ModelPrediction] = []
    for p in result.get("individual_predictions", []):
        try:
            ml_predictions.append(
                ModelPrediction(
                    ml_model=str(p.get("ml_model") or p.get("model_name")),
                    version=str(p.get("version") or p.get("model_version", "v1")),
                    prediction_score=float(p.get("prediction_score", p.get("score", 0.5))),
                    confidence=float(p.get("confidence", 0.0)),
                    processing_time_ms=float(p.get("processing_time_ms", 0.0)),
                    features_used=int(p.get("features_used", len(vector))),
                )
            )
        except Exception as e:
            logger.warning("Skipping invalid model prediction entry", error=str(e), entry=p)

    processing_time = (time.time() - start_time) * 1000

    response = PredictionResponse(
        system_id=req.sensor_data.system_id,
        prediction=prediction,
        ml_predictions=ml_predictions,
        ensemble_score=float(result.get("ensemble_score", 0.5)),
        total_processing_time_ms=float(result.get("total_processing_time_ms", processing_time)),
        features_extracted=int(len(vector)),
        cache_hit=False,
        trace_id=trace_id,
    )

    # Сохранение в кеш
    if cache_key and settings.cache_predictions:
        await cache.save_prediction(cache_key, response.model_dump())

    # Метрики
    metrics.predictions_total.labels(model_name="ensemble", prediction_type="anomaly").inc()
    metrics.inference_duration.labels(model_name="ensemble").observe(processing_time / 1000)

    return response


async def _safe_predict_item(req: PredictionRequest, ensemble: EnsembleModel, cache: CacheService) -> Union[PredictionResponse, ErrorResponse]:
    try:
        return await _predict_core(req, ensemble, cache)
    except Exception as e:
        return ErrorResponse(error=str(e), error_code="PREDICTION_FAILED", trace_id=str(uuid.uuid4()))


# ML Inference Endpoints
@router.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(
    request: PredictionRequest,
    response: Response,
    ensemble: EnsembleModel = Depends(get_ensemble_model),
    cache: CacheService = Depends(get_cache_service),
):
    return await _predict_core(request, ensemble, cache)


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    _background_tasks: BackgroundTasks,
    ensemble: EnsembleModel = Depends(get_ensemble_model),
    cache: CacheService = Depends(get_cache_service),
):
    start_time = time.time()
    trace_id = str(uuid.uuid4())

    logger.info(
        "Batch prediction started",
        batch_size=len(request.requests),
        parallel=request.parallel_processing,
        trace_id=trace_id,
    )

    if request.parallel_processing and len(request.requests) > 1:
        tasks = [_safe_predict_item(req, ensemble, cache) for req in request.requests]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    else:
        results = []
        for req in request.requests:
            results.append(await _safe_predict_item(req, ensemble, cache))

    successful = len([r for r in results if isinstance(r, PredictionResponse)])
    failed = len(results) - successful

    total_time = (time.time() - start_time) * 1000

    logger.info(
        "Batch prediction completed",
        successful=successful,
        failed=failed,
        total_time_ms=total_time,
        trace_id=trace_id,
    )

    return BatchPredictionResponse(
        results=results,
        total_processing_time_ms=total_time,
        successful_predictions=successful,
        failed_predictions=failed,
        trace_id=trace_id,
    )
