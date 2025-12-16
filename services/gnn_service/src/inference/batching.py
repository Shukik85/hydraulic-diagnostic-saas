"""Dynamic batching support for inference.

Provides:
- BatchItem: Dataclass for batch queue items
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from src.schemas import PredictionResponse
from src.schemas.requests import MinimalInferenceRequest


@dataclass
class BatchItem:
    """Item in the batch processing queue.

    Attributes:
        request: Inference request
        future: Future for returning result
        model_version: Selected model version
        enqueue_time: Timestamp when item was enqueued
        request_id: Optional request ID for tracing

    Examples:
        >>> future = asyncio.Future()
        >>> item = BatchItem(
        ...     request=my_request,
        ...     future=future,
        ...     model_version="v2",
        ...     request_id="req_123"
        ... )
    """

    request: MinimalInferenceRequest
    future: asyncio.Future[PredictionResponse]
    model_version: str
    enqueue_time: float = field(default_factory=time.time)
    request_id: str | None = None
