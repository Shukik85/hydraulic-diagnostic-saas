"""Feature engineering pipeline.

Extract features из sensor time-series:
- Statistical (mean, std, percentiles, skew, kurtosis)
- Frequency domain (FFT, PSD, dominant freq)
- Temporal (rolling windows, autocorrelation, trend)
- Hydraulic-specific (pressure ratios, efficiency)

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

if TYPE_CHECKING:
    from src.data.feature_config import FeatureConfig

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature extraction pipeline для sensor data.

    Combines multiple feature types:
    - Statistical features (11 per sensor)
    - Frequency features (num_frequencies + 2 per sensor)
    - Temporal features (11 per sensor)
    - Hydraulic-specific features (4 total)

    Args:
        config: FeatureConfig instance

    Examples:
        >>> config = FeatureConfig(use_statistical=True, num_frequencies=10)
        >>> engineer = FeatureEngineer(config)
        >>>
        >>> # Sensor data: [T, S] where T=time samples, S=sensors
        >>> data = pd.DataFrame({
        ...     "pressure_pump": [...],
        ...     "temperature": [...]
        ... })
        >>>
        >>> features = engineer.extract_all_features(data)
        >>> features.shape  # [S, total_features_per_sensor]
    """

    def __init__(self, config: FeatureConfig):
        self.config = config

        # Initialize scalers
        self.scalers: dict[str, StandardScaler | MinMaxScaler | RobustScaler] = {}

    def extract_statistical_features(self, data: pd.Series | np.ndarray) -> np.ndarray:
        """Извлечь statistical features.

        Features:
        - Mean, std, min, max
        - Percentiles (5th, 25th, 50th, 75th, 95th)
        - Skewness, kurtosis

        Args:
            data: Time-series data [T]

        Returns:
            features: Statistical features [11]

        Examples:
            >>> data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> features = engineer.extract_statistical_features(data)
            >>> features.shape  # (11,)
        """
        if isinstance(data, pd.Series):
            data = data.values

        if len(data) == 0:
            return np.zeros(11)

        features = [
            np.mean(data),
            np.std(data),
            np.min(data),
            np.max(data),
            np.median(data),
        ]

        # Percentiles
        for p in self.config.percentiles:
            features.append(np.percentile(data, p))

        # Higher-order statistics
        features.append(stats.skew(data))
        features.append(stats.kurtosis(data))

        return np.array(features, dtype=np.float32)

    def extract_frequency_features(
        self, data: pd.Series | np.ndarray, sampling_rate: float = 1.0
    ) -> np.ndarray:
        """Извлечь frequency domain features.

        Features:
        - Top N FFT magnitudes
        - Dominant frequency
        - Spectral entropy

        Args:
            data: Time-series data [T]
            sampling_rate: Sampling rate в Hz

        Returns:
            features: Frequency features [num_frequencies + 2]

        Examples:
            >>> data = np.sin(2 * np.pi * 5 * np.arange(100) / 100)  # 5 Hz signal
            >>> features = engineer.extract_frequency_features(data, sampling_rate=100)
            >>> features[0]  # Dominant frequency ~ 5.0
        """
        if isinstance(data, pd.Series):
            data = data.values

        if len(data) < 4:
            return np.zeros(self.config.num_frequencies + 2, dtype=np.float32)

        # FFT
        fft_values = rfft(data)
        fft_magnitudes = np.abs(fft_values)
        fft_freqs = rfftfreq(len(data), 1.0 / sampling_rate)

        # Top N frequencies
        top_indices = np.argsort(fft_magnitudes)[-self.config.num_frequencies :]
        top_magnitudes = fft_magnitudes[top_indices]

        # Dominant frequency
        dominant_idx = np.argmax(fft_magnitudes[1:]) + 1  # Skip DC component
        dominant_freq = fft_freqs[dominant_idx]

        # Spectral entropy
        psd = fft_magnitudes**2
        psd_normalized = psd / psd.sum()
        spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-10))

        features = np.concatenate([top_magnitudes, [dominant_freq], [spectral_entropy]])

        return features.astype(np.float32)

    def extract_temporal_features(self, data: pd.Series | np.ndarray) -> np.ndarray:
        """Извлечь temporal features.

        Features:
        - Rolling mean/std для каждого window size
        - Exponential moving average
        - Autocorrelation (lags: 1, 5, 10)
        - Linear trend slope

        Args:
            data: Time-series data [T]

        Returns:
            features: Temporal features [2 * len(window_sizes) + 5]

        Examples:
            >>> data = np.arange(100) + np.random.randn(100)  # Trending
            >>> features = engineer.extract_temporal_features(data)
            >>> features[-1]  # Trend slope ~ 1.0
        """
        series = data if isinstance(data, pd.Series) else pd.Series(data)

        if len(series) < max(self.config.window_sizes):
            return np.zeros(len(self.config.window_sizes) * 2 + 5, dtype=np.float32)

        features = []

        # Rolling statistics
        for window in self.config.window_sizes:
            rolling_mean = series.rolling(window=window).mean().iloc[-1]
            rolling_std = series.rolling(window=window).std().iloc[-1]
            features.extend([rolling_mean, rolling_std])

        # Exponential moving average
        ema = series.ewm(span=10).mean().iloc[-1]
        features.append(ema)

        # Autocorrelation
        for lag in [1, 5, 10]:
            if len(series) > lag:
                autocorr = series.autocorr(lag=lag)
                features.append(autocorr if not np.isnan(autocorr) else 0.0)
            else:
                features.append(0.0)

        # Linear trend
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series.values, deg=1)
        features.append(slope)

        return np.array(features, dtype=np.float32)

    def extract_hydraulic_features(self, data: pd.DataFrame) -> np.ndarray:
        """Извлечь hydraulic-specific features.

        Features:
        - Pressure ratio (outlet/inlet)
        - Temperature delta (max - min)
        - Flow efficiency (estimated)
        - Cavitation index (pressure-based)

        Args:
            data: DataFrame с sensor columns

        Returns:
            features: Hydraulic features [4]

        Examples:
            >>> data = pd.DataFrame({
            ...     "pressure_in": [100, 105, 102],
            ...     "pressure_out": [95, 98, 96],
            ...     "temperature": [60, 62, 61]
            ... })
            >>> features = engineer.extract_hydraulic_features(data)
            >>> features[0]  # Pressure ratio ~ 0.95
        """
        features = []

        # 1. Pressure ratio (эффективность системы)
        if "pressure_in" in data.columns and "pressure_out" in data.columns:
            p_in = data["pressure_in"].mean()
            p_out = data["pressure_out"].mean()
            pressure_ratio = p_out / (p_in + 1e-6)  # Avoid division by zero
            features.append(pressure_ratio)
        else:
            features.append(0.0)

        # 2. Temperature delta (перегрев)
        temp_cols = [col for col in data.columns if "temperature" in col.lower()]
        if temp_cols:
            temps = data[temp_cols].values.flatten()
            temp_delta = temps.max() - temps.min()
            features.append(temp_delta)
        else:
            features.append(0.0)

        # 3. Flow efficiency (simplified)
        if "flow_rate" in data.columns and "pressure_in" in data.columns:
            flow = data["flow_rate"].mean()
            pressure = data["pressure_in"].mean()
            efficiency = flow / (pressure + 1e-6)
            features.append(efficiency)
        else:
            features.append(0.0)

        # 4. Cavitation index (риск кавитации)
        if "pressure_in" in data.columns:
            p_in = data["pressure_in"]
            # Cavitation index: std(pressure) / mean(pressure)
            cavitation_index = p_in.std() / (p_in.mean() + 1e-6)
            features.append(cavitation_index)
        else:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def normalize_features(
        self, features: np.ndarray, sensor_id: str | None = None, fit: bool = False
    ) -> np.ndarray:
        """Нормализовать features.

        Args:
            features: Features array [F]
            sensor_id: Sensor identifier для per-sensor normalization
            fit: Fit scaler (для training) или transform only (inference)

        Returns:
            normalized: Normalized features [F]

        Examples:
            >>> features = np.array([100, 200, 150, 50])
            >>> normalized = engineer.normalize_features(features, fit=True)
            >>> normalized.mean()  # ~ 0.0
            >>> normalized.std()  # ~ 1.0
        """
        if self.config.normalization == "none":
            return features

        # Get or create scaler
        scaler_key = sensor_id if sensor_id else "global"

        if fit or scaler_key not in self.scalers:
            # Create scaler
            if self.config.normalization == "standardize":
                scaler = StandardScaler()
            elif self.config.normalization == "minmax":
                scaler = MinMaxScaler()
            elif self.config.normalization == "robust":
                scaler = RobustScaler()
            else:
                msg = f"Unknown normalization method: {self.config.normalization}"
                raise ValueError(msg)

            # Fit
            scaler.fit(features.reshape(-1, 1))
            self.scalers[scaler_key] = scaler
        else:
            scaler = self.scalers[scaler_key]

        # Transform
        normalized = scaler.transform(features.reshape(-1, 1)).flatten()

        return normalized.astype(np.float32)

    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Обработать missing values.

        Args:
            data: DataFrame с возможными NaN

        Returns:
            cleaned: DataFrame без NaN

        Examples:
            >>> data = pd.DataFrame({"pressure": [100, np.nan, 102, 103]})
            >>> cleaned = engineer.handle_missing_data(data)
            >>> cleaned.isna().sum().sum()  # 0
        """
        if data.isna().sum().sum() == 0:
            return data

        method = self.config.handle_missing

        if method == "ffill":
            data = data.ffill()
        elif method == "bfill":
            data = data.bfill()
        elif method == "interpolate":
            data = data.interpolate(method="linear")
        elif method == "drop":
            data = data.dropna()
        else:
            msg = f"Unknown missing data method: {method}"
            raise ValueError(msg)

        # If still NaN (e.g., all NaN column), fill with 0
        return data.fillna(0)

    def remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Удалить outliers.

        Uses z-score method: |value - mean| > threshold * std

        Args:
            data: DataFrame with sensor readings

        Returns:
            cleaned: DataFrame с заменёнными outliers (median)

        Examples:
            >>> data = pd.DataFrame({"pressure": [100, 101, 102, 500, 103]})  # 500 is outlier
            >>> cleaned = engineer.remove_outliers(data)
            >>> cleaned["pressure"].max()  # ~ 103 (outlier replaced)
        """
        threshold = self.config.outlier_threshold

        for col in data.columns:
            # Z-score
            mean = data[col].mean()
            std = data[col].std()

            if std == 0:
                continue

            z_scores = np.abs((data[col] - mean) / std)

            # Replace outliers with median
            outlier_mask = z_scores > threshold
            if outlier_mask.any():
                median_val = data[col].median()
                data.loc[outlier_mask, col] = median_val

                num_outliers = outlier_mask.sum()
                logger.debug(f"Removed {num_outliers} outliers from {col}")

        return data

    def extract_all_features(self, data: pd.DataFrame, sampling_rate: float = 1.0) -> np.ndarray:
        """Извлечь all features из sensor data.

        Args:
            data: DataFrame с sensor columns [T, S]
            sampling_rate: Sampling rate в Hz

        Returns:
            features: Feature matrix [S * features_per_sensor + hydraulic_features]

        Examples:
            >>> data = pd.DataFrame({
            ...     "pressure_pump": [...],  # T samples
            ...     "temperature": [...]     # T samples
            ... })
            >>> features = engineer.extract_all_features(data)
            >>> # Shape: [2 sensors * per_sensor_features + 4 hydraulic]
        """
        # 1. Preprocess
        data = self.handle_missing_data(data)
        data = self.remove_outliers(data)

        all_features = []

        # 2. Extract per-sensor features
        for col in data.columns:
            sensor_features = []

            # Statistical
            if self.config.use_statistical:
                stat_feats = self.extract_statistical_features(data[col])
                sensor_features.append(stat_feats)

            # Frequency
            if self.config.use_frequency:
                freq_feats = self.extract_frequency_features(data[col], sampling_rate)
                sensor_features.append(freq_feats)

            # Temporal
            if self.config.use_temporal:
                temp_feats = self.extract_temporal_features(data[col])
                sensor_features.append(temp_feats)

            # Concatenate sensor features
            if sensor_features:
                all_sensor_feats = np.concatenate(sensor_features)
                all_features.append(all_sensor_feats)

        # 3. Extract hydraulic features (global)
        if self.config.use_hydraulic:
            hydraulic_feats = self.extract_hydraulic_features(data)
            all_features.append(hydraulic_feats)

        # 4. Concatenate all
        if not all_features:
            return np.array([], dtype=np.float32)

        features = np.concatenate(all_features)

        logger.debug(f"Extracted {len(features)} features from {len(data.columns)} sensors")

        return features
