import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from src.inference import InferenceConfig, InferenceEngine
from src.schemas import PredictionResponse
from src.schemas.requests import MinimalInferenceRequest


@pytest.fixture
def mock_request() -> MinimalInferenceRequest:
    return MinimalInferenceRequest(
        equipment_id="pump_001",
        topology_id="pump_v1",
        timestamp="2025-01-01T00:00:00Z",
        sensor_readings={},
    )


@pytest.fixture
def mock_model_manager(monkeypatch):
    mm = Mock()
    mm.load_model = Mock(return_value=Mock())
    mm.warmup = Mock()
    monkeypatch.setattr("src.inference.inference_engine.ModelManager", lambda: mm)
    return mm


@pytest.mark.asyncio
async def test_inference_engine_dynamic_batching(mock_request, mock_model_manager):
    config = InferenceConfig(
        model_path="fake.ckpt",
        enable_dynamic_batching=True,
        batch_size=2,
        max_wait_ms=100.0,
    )

    engine = InferenceEngine(config)
    engine._predict_minimal_impl = AsyncMock(
        return_value=PredictionResponse(
            equipment_id="pump_001",
            health=Mock(),
            degradation=Mock(),
            anomaly=Mock(),
            inference_time_ms=10.0,
        )
    )

    task1 = asyncio.create_task(engine.predict_minimal(mock_request))
    task2 = asyncio.create_task(engine.predict_minimal(mock_request))

    await asyncio.sleep(0.15)

    await task1
    await task2

    assert engine._batch_queue is not None

    await engine.cleanup()
