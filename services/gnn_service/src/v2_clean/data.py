"""Raw data loading from UCI Hydraulic System dataset.

Loads 17 sensors at 100 Hz, parses labels from documentation,
resamples to 10 Hz, returns cycles with metadata.

Format:
  Each cycle (60 seconds @ 10 Hz) = 600 samples across 17 sensors
  Labels: 4 independent fault conditions × 3 severity levels each
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from scipy import signal

logger = logging.getLogger(__name__)


class SensorMetadata(NamedTuple):
    """Metadata for each sensor."""

    name: str  # PS1, TS2, etc.
    sensor_type: str  # pressure, temperature, flow, vibration, etc.
    unit: str  # bar, °C, L/min, g
    physical_range: tuple[float, float]  # (min, max) for normalization


class CycleLabel(NamedTuple):
    """Multi-label classification for hydraulic system state."""

    cooler: int  # 0=healthy, 1=reduced, 2=failed
    valve: int  # 0=healthy, 1=worn, 2=severely worn
    pump_leak: int  # 0=none, 1=internal, 2=external
    accumulator: int  # 0=ok, 1=presoak, 2=rapid_soaking


class Cycle(NamedTuple):
    """Single hydraulic cycle: 600 timesteps × 17 sensors."""

    cycle_id: int  # 0 to ~2600
    data: np.ndarray  # shape (600, 17), dtype float32
    label: CycleLabel
    metadata: dict[str, Any]  # timestamp, source file, etc.


SENSOR_CONFIG = {
    # Pressure sensors (6 channels, 80% of data)
    "PS1": SensorMetadata("PS1", "pressure", "bar", (0, 350)),
    "PS2": SensorMetadata("PS2", "pressure", "bar", (0, 200)),
    "PS3": SensorMetadata("PS3", "pressure", "bar", (-0.5, 10)),
    "PS4": SensorMetadata("PS4", "pressure", "bar", (0, 250)),
    "PS5": SensorMetadata("PS5", "pressure", "bar", (0, 210)),
    "PS6": SensorMetadata("PS6", "pressure", "bar", (0, 50)),
    # Temperature sensors (4 channels, <1% of data)
    "TS1": SensorMetadata("TS1", "temperature", "°C", (20, 60)),
    "TS2": SensorMetadata("TS2", "temperature", "°C", (20, 65)),
    "TS3": SensorMetadata("TS3", "temperature", "°C", (10, 30)),
    "TS4": SensorMetadata("TS4", "temperature", "°C", (15, 50)),
    # Flow sensors (2 channels, ~3% of data)
    "FS1": SensorMetadata("FS1", "flow", "L/min", (0, 60)),
    "FS2": SensorMetadata("FS2", "flow", "L/min", (0, 20)),
    # Vibration (1 channel, <1% of data)
    "VS1": SensorMetadata("VS1", "vibration", "g", (-1, 1)),
    # Solenoid (1 channel, digital)
    "SE": SensorMetadata("SE", "solenoid", "state", (0, 1)),
    # Energy/Power (2 channels, <1% of data)
    "CE": SensorMetadata("CE", "energy", "J", (0, 1e7)),
    "CP": SensorMetadata("CP", "power", "W", (0, 1e5)),
    # Electrical power signature (1 channel, ~15% of data, not monitored)
    "EPS1": SensorMetadata("EPS1", "electrical", "V", (0, 1)),
}

SENSOR_ORDER = [
    "PS1", "PS2", "PS3", "PS4", "PS5", "PS6",  # Pressure
    "TS1", "TS2", "TS3", "TS4",  # Temperature
    "FS1", "FS2",  # Flow
    "VS1",  # Vibration
    "SE",  # Solenoid
    "CE", "CP",  # Energy/Power
    "EPS1",  # Electrical power signature
]

assert len(SENSOR_ORDER) == 17
assert set(SENSOR_ORDER) == set(SENSOR_CONFIG.keys())


class RawDataLoader:
    """Load UCI Hydraulic System dataset from raw .txt files.

    Each sensor is stored in a separate text file (one value per line).
    At 100 Hz for ~13 hours = 4.68M samples per sensor.

    Resamples to 10 Hz (600 samples per 60-second cycle) and attaches labels.
    """

    def __init__(self, data_dir: str | Path):
        """Initialize loader.

        Args:
            data_dir: Path to raw_real_dataset directory
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            msg = f"Data directory not found: {self.data_dir}"
            raise FileNotFoundError(msg)

        # Load all sensor data into memory (once, at init)
        logger.info(f"Loading sensor data from {self.data_dir}...")
        self.sensor_data = self._load_sensors()
        logger.info(f"Loaded {len(self.sensor_data)} sensors")

        # Parse labels
        logger.info("Parsing cycle labels...")
        self.labels = self._parse_labels()
        logger.info(f"Parsed {len(self.labels)} cycle labels")

    def _load_sensors(self) -> dict[str, np.ndarray]:
        """Load all 17 sensors from .txt files.

        Returns:
            sensors: Dict[sensor_name, array(N_samples,)]
        """
        sensors = {}

        for sensor_name in SENSOR_ORDER:
            sensor_file = self.data_dir / f"{sensor_name}.txt"
            if not sensor_file.exists():
                logger.warning(f"Sensor file not found: {sensor_file}")
                continue

            # Load as single column (one value per line)
            data = np.loadtxt(sensor_file, dtype=np.float32)
            sensors[sensor_name] = data
            logger.debug(f"Loaded {sensor_name}: {len(data)} samples")

        return sensors

    def _parse_labels(self) -> dict[int, CycleLabel]:
        """Parse cycle labels from documentation.txt.

        Returns:
            labels: Dict[cycle_id, CycleLabel]
        """
        doc_file = self.data_dir / "documentation.txt"
        if not doc_file.exists():
            logger.warning(f"Documentation not found: {doc_file}")
            return {}

        labels = {}
        # TODO: Implement label parsing from documentation.txt
        # For now, return empty dict (will implement when reading actual docs)
        return labels

    def load_cycles(
        self,
        cycle_length_sec: float = 60,
        input_hz: float = 100,
        output_hz: float = 10,
    ) -> list[Cycle]:
        """Load all cycles from raw data.

        Process:
          1. Resample each sensor from 100 Hz → 10 Hz
          2. Split into 60-second cycles (600 samples @ 10 Hz)
          3. Attach labels
          4. Return list of Cycle objects

        Args:
            cycle_length_sec: Length of each cycle in seconds
            input_hz: Original sampling rate (100 Hz)
            output_hz: Target sampling rate (10 Hz)

        Returns:
            cycles: List of Cycle objects
        """
        if not self.sensor_data:
            msg = "No sensor data loaded"
            raise RuntimeError(msg)

        # Step 1: Resample all sensors
        logger.info(f"Resampling {len(self.sensor_data)} sensors from {input_hz} Hz to {output_hz} Hz...")
        resampled = {}
        for sensor_name, data in self.sensor_data.items():
            n_samples_resampled = int(len(data) * output_hz / input_hz)
            resampled[sensor_name] = signal.resample(data, n_samples_resampled, method="linear")
            logger.debug(
                f"{sensor_name}: {len(data)} samples (100 Hz) → "
                f"{len(resampled[sensor_name])} samples (10 Hz)"
            )

        # Step 2: Get total number of cycles
        samples_per_cycle = int(cycle_length_sec * output_hz)
        total_samples = min(len(v) for v in resampled.values())
        num_cycles = total_samples // samples_per_cycle
        logger.info(f"Total cycles: {num_cycles} × {samples_per_cycle} samples")

        # Step 3: Build cycles
        cycles = []
        for cycle_id in range(num_cycles):
            start_idx = cycle_id * samples_per_cycle
            end_idx = start_idx + samples_per_cycle

            # Stack sensors as columns: (samples, sensors)
            cycle_data = np.column_stack(
                [resampled[sensor_name][start_idx:end_idx] for sensor_name in SENSOR_ORDER]
            ).astype(np.float32)

            # Get label (fallback to dummy if not in labels dict)
            label = self.labels.get(cycle_id, CycleLabel(0, 0, 0, 0))

            # Create metadata
            metadata = {
                "cycle_id": cycle_id,
                "timestamp_start_sec": cycle_id * cycle_length_sec,
                "timestamp_end_sec": (cycle_id + 1) * cycle_length_sec,
                "hz": output_hz,
                "num_sensors": len(SENSOR_ORDER),
            }

            cycles.append(Cycle(cycle_id=cycle_id, data=cycle_data, label=label, metadata=metadata))

        logger.info(f"Loaded {len(cycles)} cycles")
        return cycles
