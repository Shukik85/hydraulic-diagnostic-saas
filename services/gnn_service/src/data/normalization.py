"""Normalization module for dynamic edge features.

Implements mixed normalization strategy tailored to each feature type:
- Log-transform + z-score for right-skewed distributions (flow)
- Z-score for features with negative values (pressure/temp delta)
- Min-max for bounded positive features (vibration, age)

Author: GNN Service Team
Python: 3.14+
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field


class NormalizationStatistics(BaseModel):
    """Statistics for edge feature normalization.

    Computed from training data and saved with model checkpoint.

    Attributes:
        # Flow rate (log-transformed)
        flow_log_mean: Mean of log(1 + flow_rate)
        flow_log_std: Std of log(1 + flow_rate)

        # Pressure drop (can be negative)
        pressure_drop_mean: Mean pressure drop
        pressure_drop_std: Std pressure drop

        # Temperature delta (can be negative)
        temp_delta_mean: Mean temperature delta
        temp_delta_std: Std temperature delta

        # Vibration (always positive)
        vibration_max: Maximum vibration level

        # Age (always positive)
        age_max: Maximum age in hours
    """

    # Flow rate statistics (log-transformed)
    flow_log_mean: float = Field(default=4.0, description="Mean of log(1 + flow)")
    flow_log_std: float = Field(default=1.0, description="Std of log(1 + flow)")

    # Pressure drop statistics
    pressure_drop_mean: float = Field(default=2.0, description="Mean ΔP (bar)")
    pressure_drop_std: float = Field(default=1.0, description="Std ΔP (bar)")

    # Temperature delta statistics
    temp_delta_mean: float = Field(default=5.0, description="Mean ΔT (°C)")
    temp_delta_std: float = Field(default=3.0, description="Std ΔT (°C)")

    # Vibration statistics
    vibration_max: float = Field(default=10.0, description="Max vibration (g)")

    # Age statistics
    age_max: float = Field(default=100000.0, description="Max age (hours)")


class EdgeFeatureNormalizer:
    """Normalizer for dynamic edge features.

    Uses mixed normalization strategy:
    1. Flow rate: log + z-score (right-skewed)
    2. Pressure drop: z-score (can be negative)
    3. Temperature delta: z-score (can be negative)
    4. Vibration: min-max [0, 1] (bounded positive)
    5. Age: min-max [0, 1] (monotonic)
    6. Maintenance score: no normalization (already [0, 1])

    Examples:
        >>> normalizer = EdgeFeatureNormalizer()
        >>>
        >>> # Fit from training data
        >>> training_features = [...]
        >>> normalizer.fit(training_features)
        >>>
        >>> # Transform for inference
        >>> normalized = normalizer.normalize_flow(115.3)
        >>>
        >>> # Save with checkpoint
        >>> normalizer.save("normalizer_stats.json")
    """

    def __init__(self, stats: NormalizationStatistics | None = None):
        """Initialize normalizer with optional pre-computed statistics.

        Args:
            stats: Pre-computed statistics (None = use defaults)
        """
        self.stats = stats or NormalizationStatistics()

    # ========================================================================
    # Flow Rate (Log-transform + Z-score)
    # ========================================================================

    def normalize_flow(self, flow_lpm: float) -> float:
        """Normalize flow rate using log-transform + z-score.

        Handles right-skewed distribution of flow rates.

        Args:
            flow_lpm: Flow rate in L/min

        Returns:
            Normalized flow rate

        Examples:
            >>> normalizer = EdgeFeatureNormalizer()
            >>> normalizer.normalize_flow(115.3)
            0.842
        """
        # Log transform (handles 0 flow gracefully)
        flow_log = np.log1p(flow_lpm)  # log(1 + x)

        # Z-score normalization
        flow_norm = (flow_log - self.stats.flow_log_mean) / self.stats.flow_log_std

        # Clip outliers to [-5, 5] standard deviations
        return np.clip(flow_norm, -5.0, 5.0)

    def denormalize_flow(self, flow_norm: float) -> float:
        """Denormalize flow rate back to L/min.

        Args:
            flow_norm: Normalized flow rate

        Returns:
            Flow rate in L/min
        """
        # Reverse z-score
        flow_log = flow_norm * self.stats.flow_log_std + self.stats.flow_log_mean

        # Reverse log transform
        flow_lpm = np.expm1(flow_log)  # exp(x) - 1

        return max(0.0, flow_lpm)  # Ensure non-negative

    # ========================================================================
    # Pressure Drop (Z-score)
    # ========================================================================

    def normalize_pressure_drop(self, dp_bar: float) -> float:
        """Normalize pressure drop using z-score.

        Can handle negative values (backflow).

        Args:
            dp_bar: Pressure drop in bar

        Returns:
            Normalized pressure drop

        Examples:
            >>> normalizer = EdgeFeatureNormalizer()
            >>> normalizer.normalize_pressure_drop(2.1)
            0.1
        """
        dp_norm = (dp_bar - self.stats.pressure_drop_mean) / self.stats.pressure_drop_std
        return np.clip(dp_norm, -5.0, 5.0)

    def denormalize_pressure_drop(self, dp_norm: float) -> float:
        """Denormalize pressure drop back to bar.

        Args:
            dp_norm: Normalized pressure drop

        Returns:
            Pressure drop in bar
        """
        return dp_norm * self.stats.pressure_drop_std + self.stats.pressure_drop_mean

    # ========================================================================
    # Temperature Delta (Z-score)
    # ========================================================================

    def normalize_temp_delta(self, dt_c: float) -> float:
        """Normalize temperature delta using z-score.

        Can handle negative values (cooling).

        Args:
            dt_c: Temperature delta in °C

        Returns:
            Normalized temperature delta

        Examples:
            >>> normalizer = EdgeFeatureNormalizer()
            >>> normalizer.normalize_temp_delta(1.5)
            -1.167
        """
        dt_norm = (dt_c - self.stats.temp_delta_mean) / self.stats.temp_delta_std
        return np.clip(dt_norm, -5.0, 5.0)

    def denormalize_temp_delta(self, dt_norm: float) -> float:
        """Denormalize temperature delta back to °C.

        Args:
            dt_norm: Normalized temperature delta

        Returns:
            Temperature delta in °C
        """
        return dt_norm * self.stats.temp_delta_std + self.stats.temp_delta_mean

    # ========================================================================
    # Vibration (Min-Max [0, 1])
    # ========================================================================

    def normalize_vibration(self, vib_g: float) -> float:
        """Normalize vibration level to [0, 1].

        Uses min-max normalization for bounded positive values.

        Args:
            vib_g: Vibration level in g

        Returns:
            Normalized vibration [0, 1]

        Examples:
            >>> normalizer = EdgeFeatureNormalizer()
            >>> normalizer.normalize_vibration(0.3)
            0.03
        """
        vib_norm = vib_g / self.stats.vibration_max
        return np.clip(vib_norm, 0.0, 1.0)

    def denormalize_vibration(self, vib_norm: float) -> float:
        """Denormalize vibration level back to g.

        Args:
            vib_norm: Normalized vibration [0, 1]

        Returns:
            Vibration level in g
        """
        return vib_norm * self.stats.vibration_max

    # ========================================================================
    # Age (Min-Max [0, 1])
    # ========================================================================

    def normalize_age(self, age_hours: float) -> float:
        """Normalize age to [0, 1].

        Uses min-max normalization for monotonic positive values.

        Args:
            age_hours: Age in hours

        Returns:
            Normalized age [0, 1]

        Examples:
            >>> normalizer = EdgeFeatureNormalizer()
            >>> normalizer.normalize_age(12500.0)
            0.125
        """
        age_norm = age_hours / self.stats.age_max
        return np.clip(age_norm, 0.0, 1.0)

    def denormalize_age(self, age_norm: float) -> float:
        """Denormalize age back to hours.

        Args:
            age_norm: Normalized age [0, 1]

        Returns:
            Age in hours
        """
        return age_norm * self.stats.age_max

    # ========================================================================
    # Maintenance Score (No normalization - already [0, 1])
    # ========================================================================

    def normalize_maintenance_score(self, score: float) -> float:
        """No normalization needed (already [0, 1]).

        Args:
            score: Maintenance score [0, 1]

        Returns:
            Same score (pass-through)
        """
        return np.clip(score, 0.0, 1.0)

    # ========================================================================
    # Batch Operations
    # ========================================================================

    def normalize_all(self, features: dict[str, float]) -> dict[str, float]:
        """Normalize all edge features at once.

        Args:
            features: Dictionary with 6 edge features

        Returns:
            Dictionary with normalized features

        Examples:
            >>> features = {
            ...     "flow_rate_lpm": 115.3,
            ...     "pressure_drop_bar": 2.1,
            ...     "temperature_delta_c": 1.5,
            ...     "vibration_level_g": 0.3,
            ...     "age_hours": 12500.0,
            ...     "maintenance_score": 0.8
            ... }
            >>> normalizer.normalize_all(features)
            {...normalized values...}
        """
        return {
            "flow_rate_lpm": self.normalize_flow(features["flow_rate_lpm"]),
            "pressure_drop_bar": self.normalize_pressure_drop(features["pressure_drop_bar"]),
            "temperature_delta_c": self.normalize_temp_delta(features["temperature_delta_c"]),
            "vibration_level_g": self.normalize_vibration(features["vibration_level_g"]),
            "age_hours": self.normalize_age(features["age_hours"]),
            "maintenance_score": self.normalize_maintenance_score(features["maintenance_score"]),
        }

    # ========================================================================
    # Fit from Training Data
    # ========================================================================

    def fit(self, training_features: list[dict[str, float]]) -> None:
        """Compute normalization statistics from training data.

        Args:
            training_features: List of feature dictionaries from training set

        Examples:
            >>> training_data = [
            ...     {"flow_rate_lpm": 100, "pressure_drop_bar": 2.0, ...},
            ...     {"flow_rate_lpm": 120, "pressure_drop_bar": 2.5, ...},
            ...     ...
            ... ]
            >>> normalizer.fit(training_data)
        """
        # Extract arrays
        flows = [f["flow_rate_lpm"] for f in training_features]
        pressures = [f["pressure_drop_bar"] for f in training_features]
        temps = [f["temperature_delta_c"] for f in training_features]
        vibrations = [f["vibration_level_g"] for f in training_features]
        ages = [f["age_hours"] for f in training_features]

        # Flow rate: log-transform then compute mean/std
        flow_logs = [np.log1p(f) for f in flows]
        self.stats.flow_log_mean = float(np.mean(flow_logs))
        self.stats.flow_log_std = float(np.std(flow_logs) + 1e-6)  # Avoid division by zero

        # Pressure drop: mean/std
        self.stats.pressure_drop_mean = float(np.mean(pressures))
        self.stats.pressure_drop_std = float(np.std(pressures) + 1e-6)

        # Temperature delta: mean/std
        self.stats.temp_delta_mean = float(np.mean(temps))
        self.stats.temp_delta_std = float(np.std(temps) + 1e-6)

        # Vibration: max
        self.stats.vibration_max = float(np.percentile(vibrations, 99))  # 99th percentile (robust)

        # Age: max
        self.stats.age_max = float(np.percentile(ages, 99))  # 99th percentile (robust)

    # ========================================================================
    # Persistence
    # ========================================================================

    def save(self, path: str | Path) -> None:
        """Save normalization statistics to JSON file.

        Typically saved alongside model checkpoint.

        Args:
            path: Path to save JSON file

        Examples:
            >>> normalizer.save("checkpoints/normalizer_v2.0.0.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.stats.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> EdgeFeatureNormalizer:
        """Load normalization statistics from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            Configured EdgeFeatureNormalizer

        Examples:
            >>> normalizer = EdgeFeatureNormalizer.load("checkpoints/normalizer_v2.0.0.json")
        """
        with open(path) as f:
            stats_dict = json.load(f)

        stats = NormalizationStatistics(**stats_dict)
        return cls(stats=stats)

    def get_stats(self) -> dict:
        """Get statistics as dictionary (for checkpoint saving).

        Returns:
            Statistics dictionary
        """
        return self.stats.model_dump()

    def load_stats(self, stats_dict: dict) -> None:
        """Load statistics from dictionary.

        Args:
            stats_dict: Statistics dictionary from checkpoint
        """
        self.stats = NormalizationStatistics(**stats_dict)


# ============================================================================
# Factory Function
# ============================================================================


def create_edge_feature_normalizer(
    stats: NormalizationStatistics | None = None,
) -> EdgeFeatureNormalizer:
    """Create EdgeFeatureNormalizer instance.

    Factory function for consistency with other modules.

    Args:
        stats: Optional pre-computed statistics

    Returns:
        Configured EdgeFeatureNormalizer

    Examples:
        >>> # With defaults
        >>> normalizer = create_edge_feature_normalizer()
        >>>
        >>> # With custom stats
        >>> stats = NormalizationStatistics(flow_log_mean=4.5, ...)
        >>> normalizer = create_edge_feature_normalizer(stats)
    """
    return EdgeFeatureNormalizer(stats=stats)
