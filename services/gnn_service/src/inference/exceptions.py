"""Inference engine exceptions.

All exceptions used throughout the inference pipeline.
"""

from __future__ import annotations


class InferenceEngineError(Exception):
    """Base exception for inference engine."""


class ModelLoadError(InferenceEngineError):
    """Model load failed."""


class GraphBuildError(InferenceEngineError):
    """Graph construction failed."""


class InferenceError(InferenceEngineError):
    """Inference execution failed."""


class TopologyNotFoundError(InferenceEngineError):
    """Topology template not found."""


class GPUOutOfMemoryError(InferenceError):
    """GPU out of memory."""


class TensorValidationError(InferenceError):
    """Tensor validation failed."""
