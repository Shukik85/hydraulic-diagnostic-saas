"""Inference module for production predictions.

Components:
- InferenceEngine - Main inference orchestrator
- DynamicGraphBuilder - Variable topology support (Phase 3)
- ModelManager - Model loading and management

Python 3.14 Features:
    - Deferred annotations
"""

from __future__ import annotations

from src.inference.dynamic_graph_builder import DynamicGraphBuilder
from src.inference.inference_engine import InferenceConfig, InferenceEngine
from src.inference.model_manager import ModelManager

__all__ = [
    "DynamicGraphBuilder",
    "InferenceConfig",
    "InferenceEngine",
    "ModelManager",
]
