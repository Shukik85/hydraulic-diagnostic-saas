# services/equipment_service/monitoring_endpoints.py
"""
Monitoring endpoints для Equipment Service.
"""
import logging
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST

import sys
sys.path.append('../shared')
from monitoring import HealthChecker, MetricsCollector, generate_metrics

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])

# Initialize
health_checker = HealthChecker("equipment-service")
metrics_collector = MetricsCollector("equipment-service")


@router.get("/health")
async def health_check():
    """
    Service health check.
    
    Returns service status and dependency checks.
    
    **Status codes**:
    - `healthy`: All systems operational
    - `degraded`: Non-critical issues detected
    - `unhealthy`: Critical failures
    """
    # Get database connection from app state
    # db = request.app.state.db
    
    health_status = await health_checker.get_health_status(
        # db=db,
        # redis=redis_client
    )
    
    return health_status


@router.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns metrics in Prometheus exposition format.
    
    **Metrics include**:
    - HTTP request counts and latency
    - Database connection pool
    - Resource usage (CPU, memory)
    - Service-specific metrics
    """
    # Update resource metrics
    metrics_collector.update_resource_metrics()
    
    # Generate Prometheus format
    metrics_data = generate_metrics()
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe для Kubernetes.
    
    Returns 200 if service is ready to accept traffic.
    """
    # Check critical dependencies
    # db_ready = await check_database()
    
    return {"status": "ready"}
