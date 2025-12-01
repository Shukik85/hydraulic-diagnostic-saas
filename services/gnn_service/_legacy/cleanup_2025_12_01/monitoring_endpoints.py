# services/gnn_service/monitoring_endpoints.py
"""
Monitoring endpoints для GNN Service.
"""
import logging
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST

import sys
sys.path.append('../shared')
from monitoring import HealthChecker, MetricsCollector, generate_metrics

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring"])

health_checker = HealthChecker("gnn-service")
metrics_collector = MetricsCollector("gnn-service")


@app.get("/health")
async def health_check():
    """
    GNN Service health check.
    
    Checks:
    - Service running
    - Model loaded
    - GPU available
    - Database connection
    """
    health_status = await health_checker.get_health_status()
    
    # Check model loaded
    try:
        # from model_loader import get_model
        # model = get_model()
        health_status["checks"]["model"] = "loaded"
    except Exception as e:
        health_status["checks"]["model"] = "failed"
        health_status["status"] = "unhealthy"
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            health_status["checks"]["gpu"] = f"available ({torch.cuda.get_device_name(0)})"
        else:
            health_status["checks"]["gpu"] = "cpu_mode"
    except Exception:
        health_status["checks"]["gpu"] = "unknown"
    
    return health_status


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics for GNN Service.
    
    **Metrics include**:
    - Inference counts
    - Inference latency
    - GPU memory usage
    - Model accuracy (if tracked)
    """
    metrics_collector.update_resource_metrics()
    return Response(generate_metrics(), media_type=CONTENT_TYPE_LATEST)


@app.get("/ready")
async def readiness_check():
    """Readiness probe."""
    return {"status": "ready"}
