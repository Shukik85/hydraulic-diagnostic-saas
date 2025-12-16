import asyncio

from src.inference.batching import BatchItem
from src.schemas.requests import MinimalInferenceRequest


def test_batch_item_creation() -> None:
    request = MinimalInferenceRequest(
        equipment_id="eq1",
        topology_id="topo1",
        timestamp="2025-01-01T00:00:00Z",
        sensor_readings={},
    )
    future: asyncio.Future = asyncio.Future()
    item = BatchItem(
        request=request,
        future=future,
        model_version="v1",
        request_id="req_123",
    )

    assert item.request is request
    assert item.future is future
    assert item.model_version == "v1"
    assert item.request_id == "req_123"
    assert isinstance(item.enqueue_time, float)
