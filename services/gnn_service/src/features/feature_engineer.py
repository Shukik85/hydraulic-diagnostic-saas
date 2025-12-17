"""Mock Feature Engineer for development.

Extracts 34-dimensional features from sensor time series:
- Statistical features (mean, std, min, max, percentiles)
- Temporal features (trends, derivatives)
- Frequency domain features
- Domain-specific hydraulic features
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Mock feature engineering for hydraulic sensors.
    
    Extracts 34 features per sensor:
    - 10 statistical features
    - 8 temporal features  
    - 8 frequency features
    - 8 domain-specific features
    
    Examples:
        >>> engineer = FeatureEngineer()
        >>> features = engineer.extract_features(sensor_data)
        >>> features.shape
        (1, 34)
    """

    def __init__(self):
        """Initialize feature engineer."""
        logger.info("🔧 Initialized FeatureEngineer (mock)")

    def extract_features(self, values: np.ndarray) -> np.ndarray:
        """Extract 34 features from time series.
        
        Args:
            values: Sensor values [T]
            
        Returns:
            Feature vector [34]
        """
        if len(values) == 0:
            return np.zeros(34)

        features = []

        # Statistical features (10)
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.median(values),
            np.percentile(values, 25),
            np.percentile(values, 75),
            np.ptp(values),  # peak-to-peak
            np.var(values),
            np.mean(np.abs(values - np.mean(values))),  # mean absolute deviation
        ])

        # Temporal features (8)
        if len(values) > 1:
            diff = np.diff(values)
            features.extend([
                np.mean(diff),  # trend
                np.std(diff),
                np.max(np.abs(diff)),  # max change
                len(np.where(diff > 0)[0]) / len(diff),  # % increasing
                self._zero_crossing_rate(values),
                self._autocorr(values, lag=1),
                self._autocorr(values, lag=5),
                self._entropy(values),
            ])
        else:
            features.extend([0.0] * 8)

        # Frequency features (8)
        if len(values) >= 8:
            fft = np.fft.rfft(values)
            power = np.abs(fft) ** 2
            freqs = np.fft.rfftfreq(len(values))
            features.extend([
                np.mean(power),
                np.std(power),
                np.max(power),
                freqs[np.argmax(power)] if len(power) > 0 else 0.0,  # dominant frequency
                np.sum(power[:len(power)//4]) / np.sum(power),  # low freq energy
                np.sum(power[len(power)//4:]) / np.sum(power),  # high freq energy
                self._spectral_centroid(power, freqs),
                self._spectral_entropy(power),
            ])
        else:
            features.extend([0.0] * 8)

        # Domain-specific hydraulic features (8)
        features.extend([
            self._pressure_stability(values),
            self._flow_consistency(values),
            self._vibration_severity(values),
            self._temperature_gradient(values),
            self._leakage_indicator(values),
            self._cavitation_risk(values),
            self._efficiency_proxy(values),
            self._degradation_indicator(values),
        ])

        return np.array(features, dtype=np.float32)

    def _zero_crossing_rate(self, x: np.ndarray) -> float:
        """Calculate zero crossing rate."""
        x_centered = x - np.mean(x)
        return np.mean(np.abs(np.diff(np.sign(x_centered)))) / 2.0

    def _autocorr(self, x: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if len(x) <= lag:
            return 0.0
        x_centered = x - np.mean(x)
        c0 = np.dot(x_centered, x_centered) / len(x)
        if c0 == 0:
            return 0.0
        c_lag = np.dot(x_centered[:-lag], x_centered[lag:]) / (len(x) - lag)
        return c_lag / c0

    def _entropy(self, x: np.ndarray, bins: int = 10) -> float:
        """Calculate Shannon entropy."""
        hist, _ = np.histogram(x, bins=bins)
        hist = hist / np.sum(hist)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def _spectral_centroid(self, power: np.ndarray, freqs: np.ndarray) -> float:
        """Calculate spectral centroid."""
        if np.sum(power) == 0:
            return 0.0
        return np.sum(freqs * power) / np.sum(power)

    def _spectral_entropy(self, power: np.ndarray) -> float:
        """Calculate spectral entropy."""
        power_norm = power / (np.sum(power) + 1e-10)
        power_norm = power_norm[power_norm > 0]
        return -np.sum(power_norm * np.log2(power_norm))

    # Domain-specific features
    def _pressure_stability(self, x: np.ndarray) -> float:
        """Pressure stability indicator."""
        return 1.0 / (1.0 + np.std(x))

    def _flow_consistency(self, x: np.ndarray) -> float:
        """Flow consistency metric."""
        if len(x) < 2:
            return 1.0
        return 1.0 / (1.0 + np.std(np.diff(x)))

    def _vibration_severity(self, x: np.ndarray) -> float:
        """Vibration severity (RMS)."""
        return np.sqrt(np.mean(x ** 2))

    def _temperature_gradient(self, x: np.ndarray) -> float:
        """Temperature change rate."""
        if len(x) < 2:
            return 0.0
        return np.mean(np.diff(x))

    def _leakage_indicator(self, x: np.ndarray) -> float:
        """Leakage detection proxy."""
        if len(x) < 10:
            return 0.0
        # Detect sustained drops
        diff = np.diff(x)
        negative_runs = np.convolve(diff < 0, np.ones(5), mode='valid')
        return np.max(negative_runs) / 5.0 if len(negative_runs) > 0 else 0.0

    def _cavitation_risk(self, x: np.ndarray) -> float:
        """Cavitation risk (high frequency spikes)."""
        if len(x) < 2:
            return 0.0
        diff = np.diff(x)
        return np.mean(np.abs(diff) > 2 * np.std(diff))

    def _efficiency_proxy(self, x: np.ndarray) -> float:
        """Efficiency proxy (signal quality)."""
        return np.mean(x) / (np.std(x) + 1e-6)

    def _degradation_indicator(self, x: np.ndarray) -> float:
        """Long-term degradation trend."""
        if len(x) < 10:
            return 0.0
        # Linear regression slope
        t = np.arange(len(x))
        slope = np.polyfit(t, x, 1)[0]
        return slope
