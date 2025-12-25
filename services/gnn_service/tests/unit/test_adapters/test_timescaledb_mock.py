"""Unit tests for TimescaleDB mock adapter.

Tests:
    - Read returns value in valid range
    - Reproducibility with same seed
    - Synthetic series generation
    - Cache management
"""

import pytest
from datetime import UTC, datetime, timedelta

from src.data.adapters.timescaledb_mock import (
    TimescaleDBMockAdapter,
    TimescaleDBRealAdapter,
    create_timescaledb_adapter,
)
from src.schemas.sensor_registry import PhysicalSensor, SensorDataSourceConfig


class TestTimescaleDBMockAdapter:
    """Test mock TimescaleDB adapter."""

    @pytest.fixture
    def mock_pressure_sensor(self):
        """Create mock pressure sensor for testing."""
        return PhysicalSensor(
            sensor_id="pt_test_001",
            sensor_type="pressure",
            measurement_range=(0.0, 400.0),
            unit="bar",
            data_source_config=SensorDataSourceConfig(source_type="timescaledb"),
        )

    @pytest.fixture
    def mock_flow_sensor(self):
        """Create mock flow sensor for testing."""
        return PhysicalSensor(
            sensor_id="fm_test_001",
            sensor_type="flow",
            measurement_range=(0.0, 200.0),
            unit="L/min",
        )

    def test_read_returns_value_in_range(self, mock_pressure_sensor):
        """Test that mock returns value within sensor range."""
        adapter = TimescaleDBMockAdapter(seed=42)
        value, timestamp = adapter.read(mock_pressure_sensor, timestamp=None)

        # Check value in valid range
        assert 0 <= value <= 400
        assert isinstance(value, float)

        # Check timestamp is recent
        assert isinstance(timestamp, datetime)
        assert abs((timestamp - datetime.now(UTC)).total_seconds()) < 60

    def test_reproducible_with_same_seed(self, mock_pressure_sensor):
        """Test that same seed produces same values."""
        adapter1 = TimescaleDBMockAdapter(seed=42)
        adapter2 = TimescaleDBMockAdapter(seed=42)

        ts = datetime.now(UTC)
        value1, _ = adapter1.read(mock_pressure_sensor, timestamp=ts)
        value2, _ = adapter2.read(mock_pressure_sensor, timestamp=ts)

        assert value1 == value2

    def test_different_seeds_produce_different_values(self, mock_pressure_sensor):
        """Test that different seeds produce different values."""
        adapter1 = TimescaleDBMockAdapter(seed=42)
        adapter2 = TimescaleDBMockAdapter(seed=123)

        ts = datetime.now(UTC)
        value1, _ = adapter1.read(mock_pressure_sensor, timestamp=ts)
        value2, _ = adapter2.read(mock_pressure_sensor, timestamp=ts)

        # Very unlikely to be exactly equal with different seeds
        assert value1 != value2

    def test_respects_sensor_range(self, mock_flow_sensor):
        """Test that generated values respect sensor measurement range."""
        adapter = TimescaleDBMockAdapter(seed=42)

        # Read multiple times
        for _ in range(10):
            value, _ = adapter.read(mock_flow_sensor, timestamp=None)
            assert 0 <= value <= 200, f"Value {value} outside range [0, 200]"

    def test_synthetic_series_caching(self, mock_pressure_sensor):
        """Test that synthetic series is cached after first generation."""
        adapter = TimescaleDBMockAdapter(seed=42)

        # First read generates series
        assert mock_pressure_sensor.sensor_id not in adapter._synthetic_cache
        adapter.read(mock_pressure_sensor, timestamp=None)
        assert mock_pressure_sensor.sensor_id in adapter._synthetic_cache

        # Cache should have multiple points
        cached_series = adapter._synthetic_cache[mock_pressure_sensor.sensor_id]
        assert len(cached_series) > 1000  # 24h * 3600s = 86400 points

    def test_clear_cache(self, mock_pressure_sensor):
        """Test cache clearing."""
        adapter = TimescaleDBMockAdapter(seed=42)

        # Generate cache
        adapter.read(mock_pressure_sensor, timestamp=None)
        assert len(adapter._synthetic_cache) > 0

        # Clear cache
        adapter.clear_cache()
        assert len(adapter._synthetic_cache) == 0

    def test_historical_query(self, mock_pressure_sensor):
        """Test querying historical timestamp."""
        adapter = TimescaleDBMockAdapter(seed=42)

        # Query 1 hour ago
        one_hour_ago = datetime.now(UTC) - timedelta(hours=1)
        value, ts = adapter.read(mock_pressure_sensor, timestamp=one_hour_ago)

        # Should return closest cached value
        assert isinstance(value, float)
        assert abs((ts - one_hour_ago).total_seconds()) < 60  # Within 1 minute

    def test_multiple_sensors_independent(self, mock_pressure_sensor, mock_flow_sensor):
        """Test that different sensors have independent series."""
        adapter = TimescaleDBMockAdapter(seed=42)

        value_pressure, _ = adapter.read(mock_pressure_sensor, timestamp=None)
        value_flow, _ = adapter.read(mock_flow_sensor, timestamp=None)

        # Different sensors, different ranges
        assert value_pressure != value_flow
        assert 0 <= value_pressure <= 400
        assert 0 <= value_flow <= 200


class TestTimescaleDBRealAdapter:
    """Test real TimescaleDB adapter (not implemented)."""

    def test_raises_not_implemented(self):
        """Test that real adapter raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not implemented yet"):
            TimescaleDBRealAdapter()


class TestCreateTimescaleDBAdapter:
    """Test factory function for adapter creation."""

    def test_creates_mock_when_enabled(self, monkeypatch):
        """Test that factory creates mock adapter when enabled."""
        # Mock settings to return mock=True
        from src import config

        monkeypatch.setattr(config.settings, "use_timescaledb_mock", True)

        adapter = create_timescaledb_adapter()
        assert isinstance(adapter, TimescaleDBMockAdapter)

    def test_attempts_real_when_disabled(self, monkeypatch):
        """Test that factory attempts real adapter when disabled."""
        from src import config

        monkeypatch.setattr(config.settings, "use_timescaledb_mock", False)

        # Should raise NotImplementedError from TimescaleDBRealAdapter
        with pytest.raises(NotImplementedError):
            create_timescaledb_adapter()
