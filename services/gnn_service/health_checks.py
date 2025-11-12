# services/gnn_service/health_checks.py
"""
Health check utilities for GNN service
"""
import torch
import logging
from db_client import get_db_client
from inference_dynamic import DynamicGNNInference
from config import api_config

logger = logging.getLogger(__name__)

async def health_check():
    db_status = None
    try:
        client = await get_db_client()
        db_status = await client.health_check()
    except Exception as e:
        db_status = False
        logger.error(f"DB health: {e}")
    model_status = False
    try:
        engine = DynamicGNNInference(api_config.model_path, api_config.metadata_path)
        model_status = True
    except Exception as e:
        logger.error(f"Model health: {e}")
    system_status = (db_status is True) and model_status
    return {
        "database": db_status,
        "model": model_status,
        "system": system_status
    }
