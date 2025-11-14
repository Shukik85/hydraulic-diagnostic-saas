#!/usr/bin/env python3
# services/rag_service/download_model.py
"""
Script для предварительной загрузки DeepSeek-R1-Distill-32B.
"""
import os
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_model():
    """
    Download DeepSeek-R1-Distill-32B from Hugging Face.
    """
    from huggingface_hub import snapshot_download
    
    model_name = os.getenv(
        "MODEL_NAME",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    )
    cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/app/models"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading {model_name}...")
    logger.info(f"Cache dir: {cache_dir}")
    
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=str(cache_dir),
            resume_download=True,
            local_files_only=False
        )
        
        logger.info("Model downloaded successfully")
        
        # Check size
        total_size = sum(
            f.stat().st_size
            for f in cache_dir.rglob('*')
            if f.is_file()
        )
        logger.info(f"Total model size: {total_size / 1024**3:.2f} GB")
        
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


if __name__ == "__main__":
    success = download_model()
    sys.exit(0 if success else 1)
