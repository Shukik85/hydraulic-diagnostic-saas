"""Semisynthetic feature engineering for hydraulic diagnostics.

Approach:
  RAW + ENGINEERED features reduce model overfitting by providing
  physical domain knowledge in feature space.

  Per sensor: 3 raw (value, min, max) + ~45 engineered
           = 48 features per node

Feature Categories:
  1. RAW (3): Last value, last 10 samples min/max
  2. STATISTICAL (8): mean, std, kurtosis, skewness over 60s window
  3. TEMPORAL (4): Trend, acceleration, rate of change, cyclicity
  4. FREQUENCY (8): FFT amplitudes at key frequencies (pump, motor, bearing)
  5. RELATIONAL (6): Correlations with physically connected sensors
  6. OPERATIONAL (4): Device state inference (pump on/off, motor load, etc.)
  7. DIAGNOSTIC (12): Fault indicators (pressure drop, temp rise, vibration, etc.)
  8. ENERGY (3): Power, work, efficiency

Total: ~48 features per sensor × 17 sensors = 816 node features
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import signal, stats
from scipy.fftpack import fft

logger = logging.getLogger(__name__)


class SemisyntheticFeatureEngineer:
    """Extract 48 semisynthetic features per sensor."""

    def __init__(
        self,
        sensor_name: str,
        sensor_type: str,
        physical_range: tuple[float, float],
        hz: float = 10,
    ):
        """Initialize feature engineer for one sensor.

        Args:
            sensor_name: PS1, TS2, etc.
            sensor_type: pressure, temperature, flow, etc.
            physical_range: (min_val, max_val) for normalization
            hz: Sampling frequency (Hz)
        """
        self.sensor_name = sensor_name
        self.sensor_type = sensor_type
        self.physical_range = physical_range
        self.hz = hz
        self.window_size = int(hz * 60)  # 600 samples for 60-second cycle

    def extract(
        self,
        values: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Extract 48 semisynthetic features from sensor timeseries.

        Args:
            values: 1D array of sensor readings (600 samples for 60s @ 10Hz)
            metadata: Optional dict with cycle_id, timestamp, etc.

        Returns:
            features: 1D array of 48 features
        """
        if len(values) < self.window_size:
            logger.warning(
                f"{self.sensor_name}: Expected {self.window_size} samples, "
                f"got {len(values)}. Padding with zeros."
            )
            values = np.pad(values, (0, self.window_size - len(values)), mode="constant")

        features = []

        # 1. RAW FEATURES (3)
        features.extend(self._extract_raw(values))

        # 2. STATISTICAL FEATURES (8)
        features.extend(self._extract_statistical(values))

        # 3. TEMPORAL FEATURES (4)
        features.extend(self._extract_temporal(values))

        # 4. FREQUENCY FEATURES (8)
        features.extend(self._extract_frequency(values))

        # 5. RELATIONAL FEATURES (6) - will be filled later with correlations
        features.extend(self._extract_relational_placeholders())

        # 6. OPERATIONAL FEATURES (4)
        features.extend(self._extract_operational(values))

        # 7. DIAGNOSTIC FEATURES (12)
        features.extend(self._extract_diagnostic(values))

        # 8. ENERGY FEATURES (3)
        features.extend(self._extract_energy(values))

        assert len(features) == 48, f"Expected 48 features, got {len(features)}"
        return np.array(features, dtype=np.float32)

    def _extract_raw(self, values: np.ndarray) -> list[float]:
        """Raw features: last value, min/max of last 10 samples."""
        return [
            values[-1],  # Last value
            np.min(values[-10:]),  # Min of last 10 samples
            np.max(values[-10:]),  # Max of last 10 samples
        ]

    def _extract_statistical(self, values: np.ndarray) -> list[float]:
        """Statistical features: mean, std, kurtosis, skewness."""
        return [
            np.mean(values),
            np.std(values),
            stats.kurtosis(values),
            stats.skew(values),
            np.percentile(values, 25),  # Q1
            np.percentile(values, 50),  # Q2 (median)
            np.percentile(values, 75),  # Q3
            np.max(values) - np.min(values),  # Range
        ]

    def _extract_temporal(self, values: np.ndarray) -> list[float]:
        """Temporal features: trend, acceleration, rate of change, cyclicity."""
        # Trend: linear fit slope
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        trend = coeffs[0]

        # Rate of change: mean absolute derivative
        diffs = np.diff(values)
        rate_of_change = np.mean(np.abs(diffs))

        # Acceleration: mean absolute 2nd derivative
        d2diffs = np.diff(diffs)
        acceleration = np.mean(np.abs(d2diffs))

        # Cyclicity: auto-correlation at expected period
        # For hydraulic pump @ 10 Hz, expect ~0.5-2 Hz component
        # Use ACF at lag=10 (1 second period)
        acf = np.correlate(values - np.mean(values), values - np.mean(values), mode="same")
        acf = acf / acf[len(acf) // 2]  # Normalize
        cyclicity = np.max(acf[len(acf) // 2 + 5 : len(acf) // 2 + 15])  # lags 5-15 samples

        return [trend, rate_of_change, acceleration, cyclicity]

    def _extract_frequency(self, values: np.ndarray) -> list[float]:
        """Frequency features: FFT amplitudes at diagnostic frequencies.

        Key frequencies for hydraulic systems:
          - Pump shaft: ~17 Hz (pump speed)
          - Motor shaft: ~12 Hz (motor speed)
          - Bearing: ~40 Hz (bearing natural frequency)
          - Cavitation: >100 Hz (high frequency)
        """
        # Compute FFT
        fft_result = np.abs(fft(values - np.mean(values)))
        freqs = np.fft.fftfreq(len(values), 1 / self.hz)

        # Extract amplitudes at diagnostic frequencies
        features = []
        for target_freq in [0.5, 1.0, 2.0, 5.0, 10.0, 17.0, 12.0, 40.0]:
            idx = np.argmin(np.abs(freqs - target_freq))
            features.append(fft_result[idx])

        return features

    def _extract_relational_placeholders(self) -> list[float]:
        """Placeholder for relational features (filled during graph construction).

        These will be correlations with adjacent sensors.
        For now, return zeros (will be updated in dataset.__getitem__).
        """
        return [0.0] * 6  # Placeholder for 6 correlation features

    def _extract_operational(self, values: np.ndarray) -> list[float]:
        """Operational features: inferred device states."""
        features = []

        if self.sensor_type == "pressure":
            # Pressure-based pump/motor state
            mean_val = np.mean(values)
            min_val, max_val = self.physical_range
            normalized = (mean_val - min_val) / (max_val - min_val) if max_val > min_val else 0
            features.append(normalized)  # Normalized pressure
            features.append(np.std(values) / (max_val - min_val) if max_val > min_val else 0)  # Pressure variability

        elif self.sensor_type == "flow":
            # Flow-based pump/motor activity
            mean_val = np.mean(values)
            min_val, max_val = self.physical_range
            normalized = (mean_val - min_val) / (max_val - min_val) if max_val > min_val else 0
            features.append(normalized)  # Normalized flow
            features.append(np.std(values) / (max_val - min_val) if max_val > min_val else 0)  # Flow variability

        elif self.sensor_type == "temperature":
            # Temperature-based thermal state
            mean_val = np.mean(values)
            min_val, max_val = self.physical_range
            normalized = (mean_val - min_val) / (max_val - min_val) if max_val > min_val else 0
            features.append(normalized)  # Normalized temperature
            features.append(np.std(values) / (max_val - min_val) if max_val > min_val else 0)  # Thermal variability

        else:
            features.append(0.0)
            features.append(0.0)

        # Generic operational features (padding)
        while len(features) < 4:
            features.append(0.0)

        return features[:4]

    def _extract_diagnostic(self, values: np.ndarray) -> list[float]:
        """Diagnostic features: fault indicators.

        - Abnormal ranges (below min or above max)
        - Sudden spikes (outliers)
        - Drift (long-term trend)
        - Noise (high-frequency content)
        - Saturation (clipping)
        - Etc.
        """
        min_val, max_val = self.physical_range
        features = []

        # 1. Below minimum range
        below_min = np.sum(values < min_val) / len(values)
        features.append(below_min)

        # 2. Above maximum range
        above_max = np.sum(values > max_val) / len(values)
        features.append(above_max)

        # 3. Number of outliers (>3 sigma)
        mean_val = np.mean(values)
        std_val = np.std(values)
        outliers = np.sum(np.abs(values - mean_val) > 3 * std_val) / len(values)
        features.append(outliers)

        # 4. Spike amplitude (max jump between samples)
        diffs = np.abs(np.diff(values))
        spike_amplitude = np.max(diffs) if len(diffs) > 0 else 0
        features.append(spike_amplitude)

        # 5. Drift magnitude (polyfit slope normalized by range)
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        drift_normalized = coeffs[0] / (max_val - min_val) if max_val > min_val else 0
        features.append(drift_normalized)

        # 6. High-frequency noise (std of high-pass filtered signal)
        # Use Butterworth high-pass at 5 Hz
        if self.hz > 10:  # Only if sampling rate allows
            b, a = signal.butter(2, 5 / (self.hz / 2), btype="high")
            filtered = signal.filtfilt(b, a, values)
            hf_noise = np.std(filtered)
            features.append(hf_noise)
        else:
            features.append(0.0)

        # 7. Saturation (values at min or max boundaries)
        saturation = (np.sum(values == min_val) + np.sum(values == max_val)) / len(values)
        features.append(saturation)

        # 8. Entropy (normalized histogram entropy)
        hist, _ = np.histogram(values, bins=20)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist[hist > 0] * np.log(hist[hist > 0]))
        features.append(entropy)

        # 9. Autocorrelation decay (how fast autocorr drops)
        acf_full = np.correlate(values - np.mean(values), values - np.mean(values), mode="full")
        acf_full = acf_full / acf_full[len(acf_full) // 2]
        acf_half = acf_full[len(acf_full) // 2 :]
        decay = np.sum(np.abs(np.diff(acf_half[:100])))  # Sum of absolute ACF changes
        features.append(decay)

        # 10-12. Padding for future diagnostic features
        features.extend([0.0, 0.0, 0.0])

        return features[:12]

    def _extract_energy(self, values: np.ndarray) -> list[float]:
        """Energy-based features (useful for power/flow sensors)."""
        # 1. Total energy (sum of squared values, proxy for work done)
        energy = np.sum(values**2)

        # 2. Mean power (average value, proxy for mean load)
        mean_power = np.mean(values)

        # 3. Peak power (max value)
        peak_power = np.max(values)

        return [energy, mean_power, peak_power]

    @staticmethod
    def feature_names() -> list[str]:
        """Return names of all 48 features (for debugging/logging)."""
        return [
            # Raw (3)
            "raw_value",
            "raw_min_10s",
            "raw_max_10s",
            # Statistical (8)
            "stat_mean",
            "stat_std",
            "stat_kurtosis",
            "stat_skewness",
            "stat_q1",
            "stat_q2",
            "stat_q3",
            "stat_range",
            # Temporal (4)
            "temp_trend",
            "temp_rate_of_change",
            "temp_acceleration",
            "temp_cyclicity",
            # Frequency (8)
            "freq_0p5hz",
            "freq_1hz",
            "freq_2hz",
            "freq_5hz",
            "freq_10hz",
            "freq_17hz",
            "freq_12hz",
            "freq_40hz",
            # Relational (6)
            "relat_corr_1",
            "relat_corr_2",
            "relat_corr_3",
            "relat_corr_4",
            "relat_corr_5",
            "relat_corr_6",
            # Operational (4)
            "oper_normalized_1",
            "oper_variability",
            "oper_state_3",
            "oper_state_4",
            # Diagnostic (12)
            "diag_below_min",
            "diag_above_max",
            "diag_outliers",
            "diag_spike_amplitude",
            "diag_drift",
            "diag_hf_noise",
            "diag_saturation",
            "diag_entropy",
            "diag_acf_decay",
            "diag_reserved_10",
            "diag_reserved_11",
            "diag_reserved_12",
            # Energy (3)
            "energy_total",
            "energy_mean",
            "energy_peak",
        ]
