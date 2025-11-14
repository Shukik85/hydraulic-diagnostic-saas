"""
Production-grade inference engine with dynamic batching, GPU memory management, 
and circuit breaker pattern.

Ключевые возможности:
- Dynamic request batching (max 50ms latency)
- Model warmup для устранения cold start
- GPU memory pooling и cleanup
- Async processing queue
- Circuit breaker для fault tolerance
- Request timeout handling
- Health monitoring
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import torch
import torch.nn as nn
from loguru import logger

from config import get_settings

# ==================== Types ====================
class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class InferenceRequest:
    """Single inference request."""
    request_id: str
    equipment_id: str
    component_features: dict[str, torch.Tensor]
    timestamp: float = field(default_factory=time.time)
    timeout: float = 30.0
    future: asyncio.Future = field(default_factory=asyncio.Future)

@dataclass
class InferenceResult:
    """Inference result with metadata."""
    request_id: str
    health_scores: dict[str, float]
    degradation_rates: dict[str, float]
    anomalies: list[dict[str, Any]]
    attention_weights: torch.Tensor | None
    inference_time_ms: float
    batch_size: int
    timestamp: float = field(default_factory=time.time)

@dataclass
class CircuitBreakerMetrics:
    """Circuit breaker metrics."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    failure_threshold: int = 5
    recovery_timeout: float = 60.0  # 1 minute
    half_open_max_calls: int = 3

# ==================== Production Inference Engine ====================
class ProductionInferenceEngine:
    """
    Production-ready inference engine with advanced features.
    - Dynamic batching for throughput optimization
    - GPU memory management
    - Circuit breaker pattern
    - Request queuing with timeout
    - Model warmup
    - Health monitoring
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        batch_size: int = 16,
        batch_timeout_ms: float = 50.0,
        queue_size: int = 100,
    ):
        """
        Initialize inference engine.
        Args:
            model: PyTorch model for inference
            device: Device to run inference on
            batch_size: Maximum batch size
            batch_timeout_ms: Max wait time for batching (ms)
            queue_size: Max queue size
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0  # Convert to seconds
        self.queue_size = queue_size
        # Request queue
        self.request_queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=queue_size)
        # Circuit breaker
        self.circuit_breaker = CircuitBreakerMetrics()
        # Background task
        self.inference_task: asyncio.Task | None = None
        self._shutdown_event = asyncio.Event()
        # Metrics
        self.total_requests = 0
        self.total_batches = 0
        self.total_inference_time = 0.0
        self.warmup_completed = False
        logger.info(
            f"InferenceEngine initialized: device={device}, "
            f"batch_size={batch_size}, timeout={batch_timeout_ms}ms"
        )
    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        try:
            logger.info("Starting inference engine...")
            await self._startup()
            self.inference_task = asyncio.create_task(self._batch_inference_loop())
            yield
        finally:
            logger.info("Shutting down inference engine...")
            await self._shutdown()
    async def _startup(self) -> None:
        self.model.to(self.device)
        self.model.eval()
        settings = get_settings()
        if settings.model.compile_model and hasattr(torch, "compile"):
            logger.info(f"Compiling model with mode={settings.model.compile_mode}")
            self.model = torch.compile(
                self.model,
                mode=settings.model.compile_mode,
                dynamic=True,
                fullgraph=True,
            )
        await self._warmup()
        logger.info("✅ Inference engine ready")
    async def _warmup(self, num_iterations: int = 10) -> None:
        logger.info(f"Warming up model ({num_iterations} iterations)...")
        dummy_batch = self._create_dummy_batch(batch_size=self.batch_size)
        warmup_times = []
        for i in range(num_iterations):
            start = time.perf_counter()
            with torch.no_grad():
                _ = self.model(dummy_batch)
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            warmup_times.append(elapsed)
            logger.debug(f"Warmup iteration {i+1}/{num_iterations}: {elapsed:.2f}ms")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        avg_time = sum(warmup_times) / len(warmup_times)
        logger.info(f"✅ Warmup complete: avg={avg_time:.2f}ms")
        self.warmup_completed = True
    def _create_dummy_batch(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        return {
            f"comp_{i}": torch.randn(batch_size, 5, 32, device=self.device)
            for i in range(10)
        }
    async def _shutdown(self) -> None:
        self._shutdown_event.set()
        if self.inference_task:
            await self.inference_task
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("✅ Inference engine shutdown complete")
    async def predict(self, equipment_id: str, component_features: dict[str, torch.Tensor], timeout: float = 30.0) -> InferenceResult:
        if not self._check_circuit_breaker():
            raise RuntimeError("Service unavailable (circuit breaker open)")
        request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            equipment_id=equipment_id,
            component_features=component_features,
            timeout=timeout,
        )
        try:
            await asyncio.wait_for(
                self.request_queue.put(request),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"Request queue full (size={self.queue_size})")
        try:
            result = await asyncio.wait_for(request.future, timeout=timeout)
            self._record_success()
            return result
        except asyncio.TimeoutError:
            self._record_failure()
            logger.error(f"Request {request.request_id} timed out after {timeout}s")
            raise
        except Exception as e:
            self._record_failure()
            logger.error(f"Request {request.request_id} failed: {e}")
            raise
    async def _batch_inference_loop(self) -> None:
        logger.info("Starting batch inference loop")
        while not self._shutdown_event.is_set():
            try:
                batch = await self._collect_batch()
                if not batch:
                    await asyncio.sleep(0.01)
                    continue
                await self._process_batch(batch)
            except Exception as e:
                logger.error(f"Batch processing error: {e}")
                await asyncio.sleep(0.1)
        logger.info("Batch inference loop stopped")
    async def _collect_batch(self) -> list[InferenceRequest]:
        batch: list[InferenceRequest] = []
        deadline = asyncio.get_event_loop().time() + self.batch_timeout
        while len(batch) < self.batch_size:
            timeout = max(0, deadline - asyncio.get_event_loop().time())
            try:
                request = await asyncio.wait_for(
                    self.request_queue.get(),
                    timeout=timeout
                )
                batch.append(request)
            except asyncio.TimeoutError:
                break
        return batch
    async def _process_batch(self, batch: list[InferenceRequest]) -> None:
        if not batch:
            return
        batch_size = len(batch)
        start_time = time.perf_counter()
        try:
            batch_features = self._collate_batch(batch)
            with torch.no_grad():
                health, degradation, attention = self.model(batch_features)
            if self.device == "cuda":
                torch.cuda.synchronize()
            inference_time = (time.perf_counter() - start_time) * 1000
            for i, request in enumerate(batch):
                result = self._create_result(
                    request=request,
                    health=health[i] if health.dim() > 1 else health,
                    degradation=degradation[i] if degradation.dim() > 1 else degradation,
                    attention=attention[i] if attention is not None else None,
                    inference_time_ms=inference_time / batch_size,
                    batch_size=batch_size,
                )
                if not request.future.done():
                    request.future.set_result(result)
            self.total_requests += batch_size
            self.total_batches += 1
            self.total_inference_time += inference_time
            logger.debug(
                f"Batch processed: size={batch_size}, "
                f"time={inference_time:.2f}ms, "
                f"avg={inference_time/batch_size:.2f}ms/req"
            )
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
    def _collate_batch(self, batch: list[InferenceRequest]) -> dict[str, torch.Tensor]:
        component_keys = batch[0].component_features.keys()
        batch_features = {}
        for comp in component_keys:
            comp_feats = [req.component_features[comp].to(self.device) for req in batch]
            batch_features[comp] = torch.stack(comp_feats, dim=0)
        return batch_features
    def _create_result(self, request: InferenceRequest, health: torch.Tensor, degradation: torch.Tensor, attention: torch.Tensor | None, inference_time_ms: float, batch_size: int,) -> InferenceResult:
        health_dict = {f"comp_{i}": float(health[i]) for i in range(health.shape[0])}
        degradation_dict = {f"comp_{i}": float(degradation[i]) for i in range(degradation.shape[0])}
        anomalies = [
            {
                "component": comp,
                "type": "low_health",
                "severity": "high" if score < 0.3 else "medium",
                "score": score
            }
            for comp, score in health_dict.items()
            if score < 0.5
        ]
        return InferenceResult(
            request_id=request.request_id,
            health_scores=health_dict,
            degradation_rates=degradation_dict,
            anomalies=anomalies,
            attention_weights=attention,
            inference_time_ms=inference_time_ms,
            batch_size=batch_size,
        )
    def _check_circuit_breaker(self) -> bool:
        cb = self.circuit_breaker
        current_time = time.time()
        if cb.state == CircuitState.OPEN:
            if current_time - cb.last_failure_time > cb.recovery_timeout:
                logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                cb.state = CircuitState.HALF_OPEN
                cb.success_count = 0
                return True
            return False
        elif cb.state == CircuitState.HALF_OPEN:
            return cb.success_count < cb.half_open_max_calls
        return True
    def _record_success(self) -> None:
        cb = self.circuit_breaker
        cb.success_count += 1
        if cb.state == CircuitState.HALF_OPEN:
            if cb.success_count >= cb.half_open_max_calls:
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED (recovered)")
                cb.state = CircuitState.CLOSED
                cb.failure_count = 0
    def _record_failure(self) -> None:
        cb = self.circuit_breaker
        cb.failure_count += 1
        cb.last_failure_time = time.time()
        if cb.state == CircuitState.HALF_OPEN:
            logger.warning("Circuit breaker: HALF_OPEN -> OPEN (still failing)")
            cb.state = CircuitState.OPEN
            cb.failure_count = 0
            cb.success_count = 0
        elif cb.state == CircuitState.CLOSED:
            if cb.failure_count >= cb.failure_threshold:
                logger.error(
                    f"Circuit breaker: CLOSED -> OPEN "
                    f"({cb.failure_count} failures)"
                )
                cb.state = CircuitState.OPEN
                cb.success_count = 0
    def get_health(self) -> dict[str, Any]:
        return {
            "status": "healthy" if self.warmup_completed else "warming_up",
            "circuit_breaker": self.circuit_breaker.state.value,
            "queue_size": self.request_queue.qsize(),
            "queue_capacity": self.queue_size,
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": (
                self.total_requests / self.total_batches
                if self.total_batches > 0 else 0
            ),
            "avg_inference_time_ms": (
                self.total_inference_time / self.total_batches
                if self.total_batches > 0 else 0
            ),
        }
    def get_metrics(self) -> dict[str, float]:
        health = self.get_health()
        return {
            "gnn_inference_requests_total": float(self.total_requests),
            "gnn_inference_batches_total": float(self.total_batches),
            "gnn_inference_queue_size": float(health["queue_size"]),
            "gnn_inference_avg_batch_size": health["avg_batch_size"],
            "gnn_inference_avg_time_ms": health["avg_inference_time_ms"],
            "gnn_circuit_breaker_failures": float(self.circuit_breaker.failure_count),
        }
__all__ = [
    "ProductionInferenceEngine",
    "InferenceRequest",
    "InferenceResult",
    "CircuitState",
]