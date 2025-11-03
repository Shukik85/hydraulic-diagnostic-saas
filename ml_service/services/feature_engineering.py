"""
Feature Engineering for Hydraulic Systems Diagnostics
Enterprise feature extraction с 25+ признаками
"""

import time
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from api.schemas import FeatureVector, SensorDataBatch
from config import FEATURE_CONFIG, settings

logger = structlog.get_logger()


class FeatureEngineer:
    """
    Enterprise feature engineering для гидравлических систем.

    Извлекает 25+ признаков:
    - Sensor features: mean, std, max, min
    - Derived features: gradients, ratios, correlations
    - Window features: trends, seasonality, stationarity
    """

    def __init__(self):
        self.feature_cache = {}
        self.sampling_frequency = settings.sampling_frequency_hz
        self.window_size = int(settings.feature_window_minutes * 60 * self.sampling_frequency)

    async def extract_features(
        self, sensor_data: SensorDataBatch, feature_groups: list[str] | None = None
    ) -> FeatureVector:
        """
        Извлечение признаков из сенсорных данных.

        Args:
            sensor_data: Пакет данных с датчиков
            feature_groups: Группы признаков для извлечения

        Returns:
            FeatureVector с извлеченными признаками
        """
        start_time = time.time()

        if feature_groups is None:
            feature_groups = ["sensor_features", "derived_features", "window_features"]

        try:
            # Преобразование в DataFrame
            df = self._readings_to_dataframe(sensor_data.readings)

            features = {}
            feature_names = []

            # Извлечение по группам
            if "sensor_features" in feature_groups:
                sensor_features = self._extract_sensor_features(df)
                features.update(sensor_features)
                feature_names.extend(FEATURE_CONFIG["sensor_features"])

            if "derived_features" in feature_groups:
                derived_features = self._extract_derived_features(df)
                features.update(derived_features)
                feature_names.extend(FEATURE_CONFIG["derived_features"])

            if "window_features" in feature_groups:
                window_features = self._extract_window_features(df)
                features.update(window_features)
                feature_names.extend(FEATURE_CONFIG["window_features"])

            # Оценка качества данных
            data_quality_score = self._calculate_data_quality(df)

            extraction_time = (time.time() - start_time) * 1000

            logger.debug(
                "Feature extraction completed",
                features_count=len(features),
                extraction_time_ms=extraction_time,
                data_quality=data_quality_score,
            )

            return FeatureVector(
                features=features,
                feature_names=feature_names,
                extraction_time_ms=extraction_time,
                data_quality_score=data_quality_score,
            )

        except Exception as e:
            logger.error("Feature extraction failed", error=str(e))
            raise

    def _readings_to_dataframe(self, readings: list[Any]) -> pd.DataFrame:
        """Преобразование sensor readings в DataFrame."""
        data = []

        for reading in readings:
            data.append(
                {
                    "timestamp": pd.to_datetime(reading.timestamp),
                    "sensor_type": reading.sensor_type,
                    "value": reading.value,
                    "unit": reading.unit,
                    "component_id": reading.component_id,
                }
            )

        df = pd.DataFrame(data)
        df = df.sort_values("timestamp")

        return df

    def _extract_sensor_features(self, df: pd.DataFrame) -> dict[str, float]:
        """Основные статистические признаки по типам датчиков."""
        features = {}

        for sensor_type in ["pressure", "temperature", "flow", "vibration"]:
            sensor_data = df[df["sensor_type"] == sensor_type]["value"]

            if len(sensor_data) > 0:
                features[f"{sensor_type}_mean"] = float(sensor_data.mean())
                features[f"{sensor_type}_std"] = float(sensor_data.std())
                features[f"{sensor_type}_max"] = float(sensor_data.max())
                features[f"{sensor_type}_min"] = float(sensor_data.min())
            else:
                # Заполнение отсутствующих значений
                features[f"{sensor_type}_mean"] = 0.0
                features[f"{sensor_type}_std"] = 0.0
                features[f"{sensor_type}_max"] = 0.0
                features[f"{sensor_type}_min"] = 0.0

        return features

    def _extract_derived_features(self, df: pd.DataFrame) -> dict[str, float]:
        """Производные признаки."""
        features = {}

        try:
            # Градиенты (производные)
            for sensor_type in ["pressure", "temperature", "flow"]:
                sensor_data = df[df["sensor_type"] == sensor_type]
                if len(sensor_data) > 1:
                    values = sensor_data["value"].values
                    gradient = np.mean(np.gradient(values))
                    features[f"{sensor_type}_gradient"] = float(gradient)
                else:
                    features[f"{sensor_type}_gradient"] = 0.0

            # Корреляции
            pressure_data = df[df["sensor_type"] == "pressure"]["value"]
            temp_data = df[df["sensor_type"] == "temperature"]["value"]

            if len(pressure_data) > 1 and len(temp_data) > 1 and len(pressure_data) == len(temp_data):
                correlation = np.corrcoef(pressure_data, temp_data)[0, 1]
                features["temp_pressure_correlation"] = float(correlation if not np.isnan(correlation) else 0.0)
            else:
                features["temp_pressure_correlation"] = 0.0

            # Отношения
            pressure_mean = features.get("pressure_mean", 1.0)
            flow_mean = features.get("flow_mean", 1.0)
            features["pressure_flow_ratio"] = float(pressure_mean / max(flow_mean, 0.001))

            # RMS для вибрации
            vibration_data = df[df["sensor_type"] == "vibration"]["value"]
            if len(vibration_data) > 0:
                features["vibration_rms"] = float(np.sqrt(np.mean(vibration_data**2)))
            else:
                features["vibration_rms"] = 0.0

            # Оценка эффективности системы
            features["system_efficiency"] = self._calculate_system_efficiency(df)

            # Rolling anomaly score
            features["anomaly_score_rolling"] = self._calculate_rolling_anomaly(df)

        except Exception as e:
            logger.warning("Some derived features failed", error=str(e))
            # Заполняем отсутствующие признаки нулями
            for feature_name in FEATURE_CONFIG["derived_features"]:
                if feature_name not in features:
                    features[feature_name] = 0.0

        return features

    def _extract_window_features(self, df: pd.DataFrame) -> dict[str, float]:
        """Признаки на основе временных окон."""
        features = {}

        try:
            # Анализ трендов
            for sensor_type in ["pressure", "temperature", "flow", "vibration"]:
                sensor_data = df[df["sensor_type"] == sensor_type]

                if len(sensor_data) > 2:
                    values = sensor_data["value"].values
                    time_indices = np.arange(len(values))

                    # Линейная регрессия для тренда
                    slope, _, r_value, _, _ = stats.linregress(time_indices, values)
                    features[f"{sensor_type}_trend_slope"] = float(slope)

                    if sensor_type == "pressure":  # Основной параметр
                        features["trend_slope"] = float(slope)

            # Автокорреляция
            pressure_data = df[df["sensor_type"] == "pressure"]["value"]
            if len(pressure_data) > 10:
                autocorr = self._calculate_autocorrelation(pressure_data.values, lag=1)
                features["autocorrelation_lag1"] = float(autocorr)
            else:
                features["autocorrelation_lag1"] = 0.0

            # Кросс-корреляция
            features["cross_correlation_max"] = self._calculate_max_cross_correlation(df)

            # Стационарность (упрощенный тест)
            features["stationarity_test"] = self._simple_stationarity_test(pressure_data)

            # Сезонность
            features["seasonality_score"] = self._detect_seasonality(pressure_data)

        except Exception as e:
            logger.warning("Some window features failed", error=str(e))
            # Заполняем отсутствующие
            for feature_name in FEATURE_CONFIG["window_features"]:
                if feature_name not in features:
                    features[feature_name] = 0.0

        return features

    def _calculate_system_efficiency(self, df: pd.DataFrame) -> float:
        """Оценка эффективности системы."""
        try:
            pressure = df[df["sensor_type"] == "pressure"]["value"]
            flow = df[df["sensor_type"] == "flow"]["value"]

            if len(pressure) > 0 and len(flow) > 0:
                # Простая оценка эффективности
                avg_pressure = pressure.mean()
                avg_flow = flow.mean()

                # Нормализация к диапазону [0, 1]
                efficiency = min(1.0, (avg_pressure * avg_flow) / 10000.0)
                return float(efficiency)

        except Exception:
            pass

        return 0.5  # Нейтральное значение

    def _calculate_rolling_anomaly(self, df: pd.DataFrame) -> float:
        """Скользящая оценка аномальности."""
        try:
            # Простой Z-score analysis
            all_values = df["value"].values
            if len(all_values) > 3:
                z_scores = np.abs(stats.zscore(all_values))
                anomaly_ratio = len(z_scores[z_scores > 2]) / len(z_scores)
                return float(min(1.0, anomaly_ratio * 2))

        except Exception:
            pass

        return 0.0

    def _calculate_autocorrelation(self, data: np.ndarray, lag: int = 1) -> float:
        """Автокорреляция с заданным лагом."""
        if len(data) <= lag:
            return 0.0

        try:
            autocorr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            return float(autocorr if not np.isnan(autocorr) else 0.0)
        except Exception:
            return 0.0

    def _calculate_max_cross_correlation(self, df: pd.DataFrame) -> float:
        """Максимальная кросс-корреляция между датчиками."""
        try:
            correlations = []
            sensor_types = ["pressure", "temperature", "flow", "vibration"]

            for i in range(len(sensor_types)):
                for j in range(i + 1, len(sensor_types)):
                    data1 = df[df["sensor_type"] == sensor_types[i]]["value"]
                    data2 = df[df["sensor_type"] == sensor_types[j]]["value"]

                    if len(data1) > 1 and len(data2) > 1 and len(data1) == len(data2):
                        corr = np.corrcoef(data1, data2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))

            return float(max(correlations) if correlations else 0.0)

        except Exception:
            return 0.0

    def _simple_stationarity_test(self, data: pd.Series) -> float:
        """Упрощенный тест стационарности."""
        if len(data) < 10:
            return 0.5

        try:
            # Простой тест: сравнение половин
            mid_point = len(data) // 2
            first_half_std = data[:mid_point].std()
            second_half_std = data[mid_point:].std()

            std_ratio = min(first_half_std, second_half_std) / max(first_half_std, second_half_std, 0.001)

            return float(std_ratio)  # Чем ближе к 1, тем стационарнее

        except Exception:
            return 0.5

    def _detect_seasonality(self, data: pd.Series) -> float:
        """Обнаружение сезонности."""
        if len(data) < 20:
            return 0.0

        try:
            # Простой FFT analysis
            fft = np.fft.fft(data.values)
            # freqs = np.fft.fftfreq(len(data))  # не используется, удалено для Ruff F841

            # Находим доминирующие частоты
            power_spectrum = np.abs(fft) ** 2
            dominant_freq_power = np.max(power_spectrum[1:])  # Исключаем DC компонент
            total_power = np.sum(power_spectrum[1:])

            seasonality_score = dominant_freq_power / max(total_power, 1.0)
            return float(min(1.0, seasonality_score))

        except Exception:
            return 0.0

    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        """Оценка качества данных."""
        if len(df) == 0:
            return 0.0

        quality_factors = []

        # Полнота данных
        expected_sensors = ["pressure", "temperature", "flow", "vibration"]
        available_sensors = df["sensor_type"].unique()
        completeness = len(available_sensors) / len(expected_sensors)
        quality_factors.append(completeness)

        # Отсутствие пропусков
        missing_values = df["value"].isna().sum()
        missing_ratio = 1.0 - (missing_values / len(df))
        quality_factors.append(missing_ratio)

        # Отсутствие выбросов
        z_scores = np.abs(stats.zscore(df["value"].dropna()))
        outlier_ratio = 1.0 - (len(z_scores[z_scores > 4]) / len(z_scores))
        quality_factors.append(outlier_ratio)

        # Итоговая оценка
        return float(np.mean(quality_factors))
