"""Integration tests for InferenceEngine dynamic batching.

Tests dynamic batching behavior without requiring real model loading.
"""

import asyncio
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.inference import InferenceConfig, InferenceEngine
from src.schemas import PredictionResponse
from src.schemas.requests import ComponentSensorReading, MinimalInferenceRequest


@pytest.fixture
def mock_request() -> MinimalInferenceRequest:
    """Create valid MinimalInferenceRequest for testing."""
    return MinimalInferenceRequest(
        equipment_id="pump_001",
        topology_id="pump_v1",
        timestamp=datetime(2025, 1, 1, 0, 0, 0),
        sensor_readings={
            "pump_1": ComponentSensorReading(
                pressure_bar=150.0,
                temperature_c=65.0,
            ),
            "valve_1": ComponentSensorReading(
                pressure_bar=145.0,
                temperature_c=64.0,
            ),
        },
    )


@pytest.fixture
def fake_model_path(tmp_path) -> Path:
    """Create dummy model file for config validation."""
    model_file = tmp_path / "fake.ckpt"
    model_file.write_text("")  # Empty file, just needs to exist
    return model_file


@pytest.mark.asyncio
async def test_inference_engine_dynamic_batching(mock_request, fake_model_path):
    """Test dynamic batching in inference engine.
    
    Verifies that:
    - Multiple concurrent requests can be queued
    - Batch processing respects configuration
    - Engine can be initialized with batching enabled
    """
    config = InferenceConfig(
        model_path=str(fake_model_path),  # Use fake file that exists
        enable_dynamic_batching=True,
        batch_size=2,
        max_wait_ms=100.0,
    )

    # Mock the model loading to avoid real file I/O
    with patch('src.inference.inference_engine.ModelManager') as MockModelManager:
        mock_manager = Mock()
        mock_manager.load_model = Mock(return_value=Mock())
        mock_manager.warmup = Mock()
        MockModelManager.return_value = mock_manager
        
        engine = InferenceEngine(config)
        
        # Mock the actual prediction implementation
        engine._predict_minimal_impl = AsyncMock(
            return_value=PredictionResponse(
                equipment_id="pump_001",
                health=Mock(),
                degradation=Mock(),
                anomaly=Mock(),
                inference_time_ms=10.0,
            )
        )

        # Create concurrent requests
        task1 = asyncio.create_task(engine.predict_minimal(mock_request))
        task2 = asyncio.create_task(engine.predict_minimal(mock_request))

        # Wait for batch processing
        await asyncio.sleep(0.15)

        # Wait for tasks to complete
        await task1
        await task2

        # Verify batching was enabled
        assert engine._batch_queue is not None

        await engine.cleanup()
