"""Mock TimescaleDB adapter for testing without real database.

Generates synthetic sensor data based on nominal values + noise.
Useful for development and testing before production deployment.

Python 3.14 features:
    - Native union types (T | None)
    - Improved random module with better type hints

Architecture:
    - TimescaleDBMockAdapter: Synthetic data generator
    - TimescaleDBRealAdapter: Placeholder for Phase 2
    - create_timescaledb_adapter(): Factory based on settings

Examples:
    >>> from src.data.adapters import create_timescaledb_adapter
    >>> adapter = create_timescaledb_adapter()
    >>> sensor = PhysicalSensor(
    ...     sensor_id="pt_001",
    ...     sensor_type="pressure",
    ...     measurement_range=(0, 400)
    ... )
    >>> value, timestamp = adapter.read(sensor, timestamp=None)
    >>> assert 0 <= value <= 400
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.schemas.sensor_registry import PhysicalSensor

from src.config import settings

__all__ = ["TimescaleDBMockAdapter", "TimescaleDBRealAdapter", "create_timescaledb_adapter"]


class TimescaleDBMockAdapter:
    """Mock TimescaleDB adapter for development/testing.

    Generates synthetic sensor data based on:
    1. Sensor nominal range (from PhysicalSensor.measurement_range)
    2. Configured noise level (from settings.mock_timescaledb_noise_level)
    3. Realistic time-series behavior (trend + noise + anomalies)

    Synthetic Model:
        value = base + trend + noise + anomaly

        base = (min + max) / 2
        trend = amplitude * sin(2π * t / period)  [slow ~1h cycle]
        noise = N(0, amplitude * noise_level)    [fast Gaussian]
        anomaly = N(0, amplitude * 0.5)          [5% probability]

    Attributes:
        seed: Random seed for reproducibility
        rng: NumPy random generator
        _synthetic_cache: Cached time-series per sensor

    Examples:
        >>> adapter = TimescaleDBMockAdapter(seed=42)
        >>> sensor = PhysicalSensor(
        ...     sensor_id="pt_001",
        ...     measurement_range=(0, 400)
        ... )
        >>> value, ts = adapter.read(sensor, timestamp=None)
        >>> assert 0 <= value <= 400
        >>>
        >>> # Reproducible with same seed
        >>> adapter2 = TimescaleDBMockAdapter(seed=42)
        >>> value2, _ = adapter2.read(sensor, ts)
        >>> assert value == value2
    """

    def __init__(self, seed: int | None = None):
        """Initialize mock adapter.

        Args:
            seed: Random seed for reproducibility (default: from settings)

        Examples:
            >>> # Use settings seed
            >>> adapter = TimescaleDBMockAdapter()
            >>>
            >>> # Custom seed for test
            >>> adapter = TimescaleDBMockAdapter(seed=123)
        """
        self.seed = seed or settings.mock_timescaledb_seed
        self.rng = np.random.default_rng(self.seed)

        # Cache for synthetic time-series (sensor_id -> list of (value, timestamp))
        self._synthetic_cache: dict[str, list[tuple[float, datetime]]] = {}

    def read(
        self, sensor: PhysicalSensor, timestamp: datetime | None = None
    ) -> tuple[float, datetime]:
        """Read synthetic sensor value.

        If timestamp is None, returns latest (now).
        If timestamp is specified, returns closest cached value.

        Args:
            sensor: Physical sensor configuration
            timestamp: Requested timestamp (None = latest/now)

        Returns:
            (value, actual_timestamp) tuple

        Examples:
            >>> adapter = TimescaleDBMockAdapter(seed=42)
            >>> sensor = PhysicalSensor(
            ...     sensor_id="pt_001",
            ...     sensor_type="pressure",
            ...     measurement_range=(0, 400)
            ... )
            >>> value, ts = adapter.read(sensor, timestamp=None)
            >>> assert isinstance(value, float)
            >>> assert isinstance(ts, datetime)
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        # Generate or retrieve cached value
        if sensor.sensor_id not in self._synthetic_cache:
            self._generate_synthetic_series(sensor)

        # Find closest cached value to timestamp
        cached_series = self._synthetic_cache[sensor.sensor_id]
        closest = min(cached_series, key=lambda x: abs((x[1] - timestamp).total_seconds()))

        return closest[0], closest[1]

    def _generate_synthetic_series(
        self,
        sensor: PhysicalSensor,
        duration_hours: int = 24,
        sampling_interval_seconds: float = 1.0,
    ) -> None:
        """Generate synthetic time-series for sensor.

        Uses realistic model:
        - Base value: midpoint of measurement range
        - Slow trend: sine wave with period ~1 hour
        - Fast noise: Gaussian noise at configured level
        - Occasional anomalies: 5% chance of outlier

        Args:
            sensor: Physical sensor to generate data for
            duration_hours: Hours of historical data to generate
            sampling_interval_seconds: Time between samples

        Examples:
            >>> adapter = TimescaleDBMockAdapter()
            >>> sensor = PhysicalSensor(sensor_id="test", measurement_range=(0, 100))
            >>> adapter._generate_synthetic_series(sensor, duration_hours=1)
            >>> assert "test" in adapter._synthetic_cache
        """
        min_val, max_val = sensor.measurement_range
        base_value = (min_val + max_val) / 2
        amplitude = (max_val - min_val) * 0.2  # ±20% variation

        now = datetime.now(UTC)
        num_samples = int(duration_hours * 3600 / sampling_interval_seconds)

        series = []
        for i in range(num_samples):
            timestamp = now - timedelta(seconds=(num_samples - i) * sampling_interval_seconds)

            # Slow trend (sine wave, period = 1 hour)
            trend = amplitude * np.sin(2 * np.pi * i / (3600 / sampling_interval_seconds))

            # Fast noise (Gaussian)
            noise = self.rng.normal(0, amplitude * settings.mock_timescaledb_noise_level)

            # Occasional anomaly (5% chance)
            if self.rng.random() < 0.05:
                anomaly = self.rng.normal(0, amplitude * 0.5)
            else:
                anomaly = 0

            # Combine
            value = base_value + trend + noise + anomaly

            # Clamp to valid range
            value = np.clip(value, min_val, max_val)

            series.append((float(value), timestamp))

        self._synthetic_cache[sensor.sensor_id] = series

    def clear_cache(self) -> None:
        """Clear synthetic data cache.

        Useful for testing or when sensor configuration changes.

        Examples:
            >>> adapter = TimescaleDBMockAdapter()
            >>> # ... generate some data ...
            >>> adapter.clear_cache()
            >>> assert len(adapter._synthetic_cache) == 0
        """
        self._synthetic_cache.clear()


class TimescaleDBRealAdapter:
    """Real TimescaleDB adapter (placeholder for Phase 2).

    TODO Phase 2:
        - Implement PostgreSQL/TimescaleDB connection pool
        - Query sensor_readings table
        - Handle time-series data efficiently
        - Connection retry logic
        - Error handling

    Raises:
        NotImplementedError: Always (not implemented yet)

    Examples:
        >>> # Will be implemented in Phase 2
        >>> # adapter = TimescaleDBRealAdapter()
        >>> # value, ts = adapter.read(sensor, timestamp)
    """

    def __init__(self):
        """Initialize real TimescaleDB adapter.

        Raises:
            NotImplementedError: Not implemented yet
        """
        raise NotImplementedError(
            "Real TimescaleDB adapter not implemented yet. "
            "Set USE_TIMESCALEDB_MOCK=true in .env for development."
        )

    def read(
        self, sensor: PhysicalSensor, timestamp: datetime | None = None
    ) -> tuple[float, datetime]:
        """Read value from TimescaleDB.

        Args:
            sensor: Physical sensor
            timestamp: Requested timestamp

        Returns:
            (value, timestamp) tuple

        Raises:
            NotImplementedError: Not implemented yet
        """
        raise NotImplementedError("Use mock adapter for now")


def create_timescaledb_adapter() -> TimescaleDBMockAdapter | TimescaleDBRealAdapter:
    """Create appropriate TimescaleDB adapter based on config.

    Reads USE_TIMESCALEDB_MOCK from settings:
        - True: Returns TimescaleDBMockAdapter (development/testing)
        - False: Returns TimescaleDBRealAdapter (production)

    Returns:
        TimescaleDBMockAdapter if USE_TIMESCALEDB_MOCK=true
        TimescaleDBRealAdapter if USE_TIMESCALEDB_MOCK=false

    Examples:
        >>> # Automatically selects mock or real based on .env
        >>> adapter = create_timescaledb_adapter()
        >>> if isinstance(adapter, TimescaleDBMockAdapter):
        ...     print("Using mock adapter for testing")
    """
    if settings.use_timescaledb_mock:
        return TimescaleDBMockAdapter()
    else:
        return TimescaleDBRealAdapter()
