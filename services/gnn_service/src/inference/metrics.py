"""Prometheus metrics for inference monitoring.

Defines all Prometheus metrics used by the inference engine:
- Request counters (total, errors)
- Histograms (duration, batch size)
- Gauges (GPU memory, queue size, cache size)
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ============================================================================
# COUNTERS
# ============================================================================

INFERENCE_REQUESTS_TOTAL = Counter(
    "gnn_inference_requests_total",
    "Total inference requests",
    ["model_version", "status"],
)

INFERENCE_ERRORS_TOTAL = Counter(
    "gnn_inference_errors_total",
    "Total inference errors",
    ["error_type"],
)

# ============================================================================
# HISTOGRAMS
# ============================================================================

INFERENCE_DURATION_SECONDS = Histogram(
    "gnn_inference_duration_seconds",
    "Inference duration in seconds",
    ["model_version"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

INFERENCE_BATCH_SIZE = Histogram(
    "gnn_inference_batch_size",
    "Actual batch size processed",
    buckets=[1, 2, 4, 8, 16, 32, 64, 128],
)

# ============================================================================
# GAUGES
# ============================================================================

MODEL_GPU_MEMORY_BYTES = Gauge(
    "gnn_model_gpu_memory_bytes",
    "GPU memory allocated by model",
    ["model_version", "device"],
)

REQUEST_QUEUE_SIZE = Gauge(
    "gnn_request_queue_size",
    "Current request queue size",
)

TOPOLOGY_CACHE_SIZE = Gauge(
    "gnn_topology_cache_size",
    "Number of cached topologies",
)
