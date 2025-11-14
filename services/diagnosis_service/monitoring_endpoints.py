# services/diagnosis_service/monitoring_endpoints.py
"""
Monitoring endpoints для Diagnosis Service.
"""
import logging
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST

import sys
sys.path.append('../shared')
from monitoring import HealthChecker, MetricsCollector, generate_metrics

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])

health_checker = HealthChecker("diagnosis-service")
metrics_collector = MetricsCollector("diagnosis-service")


@router.get("/health")
async def health_check():
    """
    Health check для diagnosis orchestrator.
    
    Checks:
    - TimescaleDB connection
    - GNN Service availability
    - RAG Service availability
    - Redis/Celery
    """
    health_status = await health_checker.get_health_status()
    
    # Additional checks
    health_status["checks"]["gnn_service"] = await check_gnn_service()
    health_status["checks"]["rag_service"] = await check_rag_service()
    
    # Overall status
    if health_status["checks"]["gnn_service"] == "failed":
        health_status["status"] = "degraded"
    
    return health_status


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics.
    
    **Diagnosis-specific metrics**:
    - Total diagnoses run
    - Average diagnosis time
    - Success/failure rates
    - Queue length
    """
    metrics_collector.update_resource_metrics()
    return Response(generate_metrics(), media_type=CONTENT_TYPE_LATEST)


@router.get("/ready")
async def readiness_check():
    """Readiness probe."""
    return {"status": "ready"}


async def check_gnn_service() -> str:
    """Check GNN Service availability."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://gnn-service:8002/health",
                timeout=5.0
            )
            return "ok" if response.status_code == 200 else "degraded"
    except Exception:
        return "failed"


async def check_rag_service() -> str:
    """Check RAG Service availability."""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://rag-service:8004/health",
                timeout=5.0
            )
            return "ok" if response.status_code == 200 else "degraded"
    except Exception:
        return "failed"
