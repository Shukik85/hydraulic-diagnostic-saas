"""FastAPI application для GNN Service.

Endpoints:
- POST /predict - single inference
- POST /batch_predict - fleet batch inference
- GET /health - health check
- GET /metrics - Prometheus metrics
"""

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest

from app.auth import verify_api_key
from app.config import get_settings
from app.models.inference import InferenceEngine
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Prometheus metrics
inference_counter = Counter(
    "gnn_inference_total",
    "Total number of GNN inferences",
    ["type"],  # single/batch
)
inference_duration = Histogram(
    "gnn_inference_duration_seconds",
    "GNN inference duration in seconds",
    ["type"],
)
anomaly_counter = Counter(
    "gnn_anomalies_detected",
    "Total number of anomalies detected",
)

# Global inference engine
inference_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager для загрузки модели при старте."""
    global inference_engine
    
    logger.info("Loading GNN model", model_path=settings.model_path)
    inference_engine = InferenceEngine(settings.model_path, settings.device)
    await inference_engine.load_model()
    logger.info("GNN model loaded successfully")
    
    yield
    
    logger.info("Shutting down GNN service")


app = FastAPI(
    title="GNN Service",
    description="System-level Hydraulic Diagnostics with T-GAT",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    model_loaded = inference_engine is not None and inference_engine.model_loaded
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        device=settings.device,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
async def predict(
    request: PredictionRequest,
    api_key: str = verify_api_key,
) -> PredictionResponse:
    """Single equipment inference.
    
    Args:
        request: Graph data с node_features, edge_index, edge_attr
        api_key: Internal API key (from dependency)
    
    Returns:
        Prediction с anomaly score и explainability
    """
    if inference_engine is None or not inference_engine.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    
    inference_counter.labels(type="single").inc()
    
    with inference_duration.labels(type="single").time():
        try:
            result = await inference_engine.predict(
                node_features=request.node_features,
                edge_index=request.edge_index,
                edge_attr=request.edge_attr,
                component_names=request.component_names,
            )
        except Exception as e:
            logger.error("Inference failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Inference failed: {e}",
            )
    
    if result["prediction"] == 1:
        anomaly_counter.inc()
    
    return PredictionResponse(**result)


@app.post(
    "/batch_predict",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    status_code=status.HTTP_200_OK,
)
async def batch_predict(
    request: BatchPredictionRequest,
    api_key: str = verify_api_key,
) -> BatchPredictionResponse:
    """Batch inference для fleet management.
    
    Args:
        request: List of graphs для multiple equipment
        api_key: Internal API key
    
    Returns:
        List of predictions
    """
    if inference_engine is None or not inference_engine.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    
    inference_counter.labels(type="batch").inc()
    
    with inference_duration.labels(type="batch").time():
        try:
            results = await inference_engine.batch_predict(request.graphs)
        except Exception as e:
            logger.error("Batch inference failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch inference failed: {e}",
            )
    
    # Count anomalies
    anomaly_count = sum(1 for r in results if r["prediction"] == 1)
    anomaly_counter.inc(anomaly_count)
    
    return BatchPredictionResponse(predictions=results)


@app.get("/metrics", tags=["Monitoring"])
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")
