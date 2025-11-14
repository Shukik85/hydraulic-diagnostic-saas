#!/usr/bin/env python3
# services/rag_service/download_model.py
"""
Скрипт для предварительной загрузки DeepSeek-R1-Distill-32B.

UPDATED: Config-based, structured logging.
"""
import sys
from pathlib import Path
import structlog

from config import config

logger = structlog.get_logger()


def download_model():
    """
    Download DeepSeek-R1-Distill-32B from Hugging Face.
    
    Uses config for model name and path.
    
    Returns:
        bool: True if successful
    """
    from huggingface_hub import snapshot_download
    
    model_name = config.MODEL_NAME
    cache_dir = Path(config.MODEL_PATH)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(
        "model_download_started",
        model=model_name,
        cache_dir=str(cache_dir)
    )
    
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False
        )
        
        # Check downloaded size
        total_size = sum(
            f.stat().st_size
            for f in cache_dir.rglob('*')
            if f.is_file()
        )
        size_gb = total_size / 1024**3
        
        logger.info(
            "model_download_completed",
            model=model_name,
            size_gb=f"{size_gb:.2f}"
        )
        
        return True
        
    except Exception as e:
        logger.error(
            "model_download_failed",
            model=model_name,
            error=str(e),
            error_type=type(e).__name__
        )
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
