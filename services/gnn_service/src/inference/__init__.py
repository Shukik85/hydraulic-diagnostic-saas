"""Inference module for production deployment.

Components:
- ModelManager - Model loading, caching, versioning
- InferenceEngine - Batch inference, GPU optimization
- FastAPI integration - REST API endpoints

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

from src.inference.model_manager import ModelManager, ModelConfig
from src.inference.inference_engine import InferenceEngine, InferenceConfig

__all__ = [
    "ModelManager",
    "ModelConfig",
    "InferenceEngine",
    "InferenceConfig",
]
