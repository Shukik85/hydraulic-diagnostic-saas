"""Prometheus metrics for diagnosis service"""

from prometheus_client import Counter, Histogram, Gauge, Info
import time
from functools import wraps
from typing import Callable
import logging

logger = logging.getLogger(__name__)

# Request metrics
diagnosis_requests_total = Counter(
    'diagnosis_requests_total',
    'Total diagnosis requests',
    ['status', 'model_version']
)

diagnosis_duration_seconds = Histogram(
    'diagnosis_duration_seconds',
    'Diagnosis processing duration',
    ['stage'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

# GNN metrics
gnn_inference_duration_seconds = Histogram(
    'gnn_inference_duration_seconds',
    'GNN inference latency',
    ['model_version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

gnn_anomaly_score = Histogram(
    'gnn_anomaly_score',
    'GNN anomaly score distribution',
    ['model_version'],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

gnn_confidence = Histogram(
    'gnn_confidence',
    'GNN prediction confidence',
    ['model_version'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

# RAG metrics
rag_generation_duration_seconds = Histogram(
    'rag_generation_duration_seconds',
    'RAG interpretation generation latency',
    ['model_version'],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
)

rag_tokens_used = Histogram(
    'rag_tokens_used',
    'RAG tokens consumed per request',
    ['model_version'],
    buckets=[100, 500, 1000, 1500, 2000, 3000, 5000]
)

rag_confidence = Histogram(
    'rag_confidence',
    'RAG interpretation confidence',
    ['model_version'],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0]
)

# Error metrics
diagnosis_errors_total = Counter(
    'diagnosis_errors_total',
    'Total diagnosis errors',
    ['error_type', 'stage']
)

# Active sessions
active_diagnosis_sessions = Gauge(
    'active_diagnosis_sessions',
    'Number of active diagnosis sessions'
)

# Model info
model_info = Info(
    'model_info',
    'Current model versions in production'
)

def track_diagnosis_request(stage: str):
    """
    Decorator для отслеживания длительности diagnosis stage
    
    Usage:
        @track_diagnosis_request('gnn_inference')
        async def process_gnn(...):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            error_occurred = False
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                diagnosis_errors_total.labels(
                    error_type=type(e).__name__,
                    stage=stage
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                diagnosis_duration_seconds.labels(stage=stage).observe(duration)
                
                status = 'error' if error_occurred else 'success'
                logger.info(
                    f"Stage '{stage}' completed",
                    extra={
                        'stage': stage,
                        'duration_seconds': duration,
                        'status': status
                    }
                )
        
        return wrapper
    return decorator

def record_gnn_metrics(model_version: str, inference_time_ms: float, anomaly_score: float, confidence: float):
    """Записать метрики GNN inference"""
    gnn_inference_duration_seconds.labels(model_version=model_version).observe(inference_time_ms / 1000)
    gnn_anomaly_score.labels(model_version=model_version).observe(anomaly_score)
    gnn_confidence.labels(model_version=model_version).observe(confidence)

def record_rag_metrics(model_version: str, processing_time_ms: float, tokens_used: int, confidence: float):
    """Записать метрики RAG generation"""
    rag_generation_duration_seconds.labels(model_version=model_version).observe(processing_time_ms / 1000)
    rag_tokens_used.labels(model_version=model_version).observe(tokens_used)
    rag_confidence.labels(model_version=model_version).observe(confidence)

def record_diagnosis_result(status: str, model_version: str):
    """Записать результат diagnosis"""
    diagnosis_requests_total.labels(status=status, model_version=model_version).inc()

def update_model_info(gnn_version: str, rag_version: str):
    """Обновить информацию о текущих версиях моделей"""
    model_info.info({
        'gnn_version': gnn_version,
        'rag_version': rag_version
    })
