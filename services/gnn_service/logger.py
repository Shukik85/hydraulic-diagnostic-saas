# services/gnn_service/logger.py
"""
Structured JSON logger for production GNN service
"""
import logging
import sys
from pathlib import Path


def setup_logging(log_dir="logs", log_level="INFO", log_format="json"):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()
    stream = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    # (file handler etc. можно добавить по необходимости)
