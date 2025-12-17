"""Mock TimescaleDB connector for development.

Provides synthetic sensor data for testing without real database.
Generates realistic time series with:
- Missing values (0-50% per sensor)
- Degradation trends
- Physical correlations
- Multiple sensor types
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)


class TimescaleConnector:
    """Mock TimescaleDB connection for development/testing.
    
    Examples:
        >>> connector = TimescaleConnector(seed=42)
        >>> df = await connector.fetch_sensor_data(
        ...     equipment_id="pump_001",
        ...     start_time="2024-01-01T00:00:00",
        ...     end_time="2024-01-02T00:00:00",
        ... )
        >>> print(df.shape)
        (86400, 3)  # ~10 sensors × 8640 timesteps
    """

    def __init__(self, seed: int = 42):
        """Initialize mock connector.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        logger.info("🔧 Initialized TimescaleConnector (mock) with seed=%d", seed)

    async def fetch_sensor_data(
        self,
        equipment_id: str,
        start_time: str,
        end_time: str,
    ) -> pl.DataFrame:
        """Generate synthetic sensor data.

        Args:
            equipment_id: Equipment identifier
            start_time: Start timestamp (ISO format)
            end_time: End timestamp (ISO format)

        Returns:
            Polars DataFrame with columns [timestamp, component_id, value]
        """
        # Parse timestamps
        start_dt = datetime.fromisoformat(start_time.replace("Z", ""))
        end_dt = datetime.fromisoformat(end_time.replace("Z", ""))

        # Generate timestamps (10 second intervals)
        timestamps = pd.date_range(start=start_dt, end=end_dt, freq="10S")
        n_timesteps = len(timestamps)

        # Define sensors (matches GraphTopology)
        sensor_ids = [
            f"{equipment_id}_pump",
            f"{equipment_id}_valve_main",
            f"{equipment_id}_valve_relief",
            f"{equipment_id}_pipe_supply",
            f"{equipment_id}_pipe_return",
            f"{equipment_id}_tank",
            f"{equipment_id}_filter",
            f"{equipment_id}_actuator",
            f"{equipment_id}_cooler",
            f"{equipment_id}_accumulator",
        ]

        # Generate synthetic data
        df_data = []
        base_noise = 0.1

        for sensor_id in sensor_ids:
            series = self._generate_synthetic_series(
                n_timesteps=n_timesteps,
                sensor_id=sensor_id,
                base_noise=base_noise,
                degradation_start=int(0.7 * n_timesteps),
            )

            # Add missing values (0-50%)
            missing_ratio = np.random.uniform(0.0, 0.5)
            n_missing = int(n_timesteps * missing_ratio)
            if n_missing > 0:
                missing_indices = np.random.choice(n_timesteps, size=n_missing, replace=False)
                series[missing_indices] = np.nan

            # Create records
            for t, value in enumerate(series):
                if not np.isnan(value):  # Only non-missing values
                    df_data.append({
                        "timestamp": timestamps[t],
                        "component_id": sensor_id,
                        "value": float(value),
                    })

        # Convert to polars
        df = pl.from_dicts(df_data)

        # Remove duplicates (if any)
        df = df.unique(subset=["timestamp", "component_id"], keep="first")

        logger.info(
            "✅ Generated synthetic data for %s: %d rows, %d sensors, %d timesteps",
            equipment_id,
            len(df),
            len(sensor_ids),
            n_timesteps,
        )

        return df

    def _generate_synthetic_series(
        self,
        n_timesteps: int,
        sensor_id: str,
        base_noise: float,
        degradation_start: int,
    ) -> np.ndarray:
        """Generate realistic time series for a sensor.
        
        Args:
            n_timesteps: Number of time points
            sensor_id: Sensor identifier
            base_noise: Noise level
            degradation_start: When degradation begins
            
        Returns:
            Time series array
        """
        t = np.arange(n_timesteps)
        noise = np.random.normal(0, base_noise, n_timesteps)

        # Pump: vibration increases with wear
        if "pump" in sensor_id:
            base = 0.5 + 0.3 * np.sin(0.01 * t)
            degradation = np.where(t > degradation_start, (t - degradation_start) * 0.002, 0)
            return base + degradation + noise

        # Valves: pressure fluctuations
        elif "valve" in sensor_id:
            return 3.0 + 0.5 * np.sin(0.02 * t) + 0.001 * t + noise

        # Pipes: flow degradation (blockage)
        elif "pipe" in sensor_id:
            base = 10.0 - 0.0005 * t
            fluct = 0.5 * np.sin(0.03 * t + 1)
            return base + fluct + noise

        # Tank: level oscillation
        elif "tank" in sensor_id:
            return 50 + 20 * np.sin(0.005 * t) + noise

        # Filter: differential pressure increases (clogging)
        elif "filter" in sensor_id:
            return 0.2 + 0.0003 * t + noise

        # Actuator: position cycling
        elif "actuator" in sensor_id:
            return 50 + 30 * np.sin(0.008 * t) + noise

        # Cooler: temperature rise
        elif "cooler" in sensor_id:
            return 60 + 0.001 * t + 5 * np.sin(0.01 * t) + noise

        # Accumulator: pressure variation
        elif "accumulator" in sensor_id:
            return 2.5 + 0.5 * np.sin(0.012 * t) + noise

        # Default
        else:
            return 5.0 + 0.5 * np.sin(0.01 * t) + noise

    async def health_check(self) -> bool:
        """Check connection health.
        
        Returns:
            True (always healthy for mock)
        """
        logger.info("✅ Mock health check: OK")
        return True

    async def close(self) -> None:
        """Close connection."""
        logger.info("🔒 Mock connection closed")
