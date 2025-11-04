"""
Feature Engineering for Hydraulic Systems Diagnostics
Enterprise feature extraction с 25+ признаками
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from api.schemas import FeatureVector, SensorDataBatch
from config import FEATURE_CONFIG, settings

logger = structlog.get_logger()


# Enterprise: flexible sensor type mapping and unit conversion
SENSOR_TYPE_MAPPING: dict[str, list[str]] = {
    "pressure": ["pressure", "press", "hydraulic_pressure", "ps", "ps1", "ps2", "ps3", "ps4", "ps5", "ps6"],
    "temperature": ["temperature", "temp", "ts", "ts1", "ts2", "ts3", "ts4", "coolant_temp"],
    "flow": ["flow", "volume_flow", "flow_rate", "lpm", "fs", "fs1", "fs2"],
    "vibration": ["vibration", "vibr", "acceleration", "shake", "vs", "vs1"],
    "motor_power": ["eps1", "motor_power", "power", "watt"],
    "cooling_efficiency": ["ce", "cooling_efficiency"],
    "cooling_power": ["cp", "cooling_power"],
    "system_efficiency": ["se", "system_efficiency"],
}

UNIT_CONVERSIONS = {
    "pressure": {
        "bar": 1.0,
        "psi": 0.0689476,  # psi->bar
        "kpa": 0.01,       # kPa->bar
        "mpa": 10.0,       # MPa->bar
    },
    "temperature": {
        "c": lambda x: x,
        "f": lambda x: (x - 32) * 5.0 / 9.0,
        "k": lambda x: x - 273.15,
    },
    "flow": {
        "l/min": 1.0,
        "lpm": 1.0,
        "m3/h": 16.6667,  # m3/h -> l/min
    },
    "vibration": {
        "mm/s": 1.0,
    },
}

INDUSTRIAL_DEFAULTS = {
    "temperature_mean": 40.0,
    "flow_mean": 10.0,
    "vibration_rms": 0.5,
}


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

    def _canonical_type(self, raw_type: str) -> str:
        rt = (raw_type or "").lower()
        for canonical, aliases in SENSOR_TYPE_MAPPING.items():
            if rt in aliases:
                return canonical
        # быстрые хелперы для UCI префиксов (ps1..ps6, ts1..ts4, fs1..fs2, vs1, ce, cp, se, eps1)
        if rt.startswith("ps"):
            return "pressure"
        if rt.startswith("ts"):
            return "temperature"
        if rt.startswith("fs"):
            return "flow"
        if rt.startswith("vs"):
            return "vibration"
        return rt  # as-is

    def _convert_unit(self, value: float, unit: str | None, canonical: str) -> float:
        if unit is None:
            return float(value)
        u = unit.lower()
        conv = UNIT_CONVERSIONS.get(canonical, {})
        if u in conv:
            factor = conv[u]
            return float(factor(value) if callable(factor) else value * factor)
        return float(value)

    async def extract_features(
        self, sensor_data: SensorDataBatch, feature_groups: list[str] | None = None
    ) -> FeatureVector:
        start_time = time.time()
        if feature_groups is None:
            feature_groups = ["sensor_features", "derived_features", "window_features"]

        # Build DataFrame with canonical sensor types and unit conversion
        data = []
        for r in sensor_data.readings:
            ctype = self._canonical_type(r.sensor_type)
            val = self._convert_unit(r.value, getattr(r, "unit", None), ctype)
            data.append({
                "timestamp": pd.to_datetime(r.timestamp),
                "sensor_type": ctype,
                "value": val,
                "unit": getattr(r, "unit", None),
                "component_id": r.component_id,
            })
        df = pd.DataFrame(data).sort_values("timestamp")

        # Диагностика входа
        try:
            logger.info(
                "FeatureEngineer input",
                system_id=str(sensor_data.system_id),
                sensor_types=list(map(str, df["sensor_type"].unique())),
                total=len(df),
            )
        except Exception:
            pass

        features: dict[str, float] = {}
        feature_names: list[str] = []

        # Группы
        if "sensor_features" in feature_groups:
            features.update(self._extract_sensor_features(df))
            feature_names.extend(FEATURE_CONFIG["sensor_features"])

        if "derived_features" in feature_groups:
            derived = self._extract_derived_features(df)
            features.update(derived)
            feature_names.extend(FEATURE_CONFIG["derived_features"])

        if "window_features" in feature_groups:
            win = self._extract_window_features(df)
            features.update(win)
            feature_names.extend(FEATURE_CONFIG["window_features"])

        # Качество данных
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

    def _extract_sensor_features(self, df: pd.DataFrame) -> dict[str, float]:
        features: dict[str, float] = {}

        def safe_stats(series: pd.Series, name_prefix: str) -> None:
            if len(series) > 0:
                features[f"{name_prefix}_mean"] = float(series.mean())
                features[f"{name_prefix}_std"] = float(series.std())
                features[f"{name_prefix}_max"] = float(series.max())
                features[f"{name_prefix}_min"] = float(series.min())
            else:
                # enterprise defaults
                defaults = {
                    f"{name_prefix}_mean": INDUSTRIAL_DEFAULTS.get(f"{name_prefix}_mean", 0.0),
                    f"{name_prefix}_std": 0.0,
                    f"{name_prefix}_max": 0.0,
                    f"{name_prefix}_min": 0.0,
                }
                features.update(defaults)

        # Канонические ряды
        pressure = df[df["sensor_type"] == "pressure"]["value"]
        temperature = df[df["sensor_type"] == "temperature"]["value"]
        flow = df[df["sensor_type"] == "flow"]["value"]
        vibration = df[df["sensor_type"] == "vibration"]["value"]
        motor_power = df[df["sensor_type"] == "motor_power"]["value"]
        ce = df[df["sensor_type"] == "cooling_efficiency"]["value"]
        cp = df[df["sensor_type"] == "cooling_power"]["value"]
        se = df[df["sensor_type"] == "system_efficiency"]["value"]

        safe_stats(pressure, "pressure")
        safe_stats(temperature, "temperature")
        safe_stats(flow, "flow")

        # Вибрации — RMS
        if len(vibration) > 0:
            features["vibration_rms"] = float(np.sqrt(np.mean(vibration.values ** 2)))
        else:
            features["vibration_rms"] = INDUSTRIAL_DEFAULTS.get("vibration_rms", 0.0)

        # Энергоэффективность и охлаждение
        features["system_efficiency"] = float(se.mean()) if len(se) > 0 else 0.0
        features["cooling_efficiency"] = float(ce.mean()) if len(ce) > 0 else 0.0
        features["cooling_power"] = float(cp.mean()) if len(cp) > 0 else 0.0
        features["motor_power_mean"] = float(motor_power.mean()) if len(motor_power) > 0 else 0.0

        return features

    def _extract_derived_features(self, df: pd.DataFrame) -> dict[str, float]:
        features: dict[str, float] = {}

        try:
            # Градиенты (упрощенные)
            def gradient(series: pd.Series) -> float:
                if len(series) > 1:
                    return float(np.mean(np.gradient(series.values)))
                return 0.0

            pressure = df[df["sensor_type"] == "pressure"]["value"]
            temperature = df[df["sensor_type"] == "temperature"]["value"]
            flow = df[df["sensor_type"] == "flow"]["value"]

            features["pressure_gradient"] = gradient(pressure)
            features["temperature_gradient"] = gradient(temperature)
            features["flow_gradient"] = gradient(flow)

            # Корреляция температура↔давление (на приведенных длинах)
            p = pressure.values
            t = temperature.values
            n = min(len(p), len(t))
            if n > 2:
                corr = np.corrcoef(p[:n], t[:n])[0, 1]
                features["temp_pressure_correlation"] = float(0.0 if np.isnan(corr) else corr)
            else:
                features["temp_pressure_correlation"] = 0.0

            # Отношение давления к потоку
            p_mean = float(pressure.mean()) if len(pressure) > 0 else 0.0
            f_mean = float(flow.mean()) if len(flow) > 0 else 0.0
            features["pressure_flow_ratio"] = float(p_mean / max(f_mean, 1e-3))

            # Rolling anomaly proxy → убран, не нужен в canonical-25 напрямую
        except Exception as e:
            logger.warning("Derived features warning", error=str(e))
            for name in ["pressure_gradient", "temperature_gradient", "flow_gradient", "temp_pressure_correlation", "pressure_flow_ratio"]:
                features.setdefault(name, 0.0)

        return features

    def _extract_window_features(self, df: pd.DataFrame) -> dict[str, float]:
        features: dict[str, float] = {}
        try:
            # Тренд по давлению
            pressure = df[df["sensor_type"] == "pressure"]["value"]
            if len(pressure) > 2:
                values = pressure.values
                idx = np.arange(len(values))
                slope, _, _, _, _ = stats.linregress(idx, values)
                features["trend_slope"] = float(slope)
            else:
                features["trend_slope"] = 0.0

            # Автокорреляция лаг-1 по давлению
            if len(pressure) > 10:
                arr = pressure.values
                autocorr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
                features["autocorrelation_lag1"] = float(0.0 if np.isnan(autocorr) else autocorr)
            else:
                features["autocorrelation_lag1"] = 0.0

            # Сезонность по температуре (упрощенно: отношение max spectral power к total)
            temperature = df[df["sensor_type"] == "temperature"]["value"]
            if len(temperature) > 20:
                fft = np.fft.fft(temperature.values)
                power = np.abs(fft) ** 2
                dom = float(np.max(power[1:])) if len(power) > 1 else 0.0
                tot = float(np.sum(power[1:])) if len(power) > 1 else 1.0
                features["seasonality_score"] = float(min(1.0, dom / max(tot, 1e-6)))
            else:
                features["seasonality_score"] = 0.0
        except Exception as e:
            logger.warning("Window features warning", error=str(e))
            for name in ["trend_slope", "autocorrelation_lag1", "seasonality_score"]:
                features.setdefault(name, 0.0)

        return features

    def _calculate_data_quality(self, df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        quality = []
        expected = ["pressure", "temperature", "flow", "vibration"]
        available = df["sensor_type"].unique()
        completeness = len([s for s in expected if s in available]) / len(expected)
        quality.append(completeness)
        missing = df["value"].isna().sum()
        quality.append(1.0 - (missing / max(len(df), 1)))
        try:
            z = np.abs(stats.zscore(df["value"].dropna()))
            outlier_ratio = 1.0 - (len(z[z > 4]) / max(len(z), 1))
        except Exception:
            outlier_ratio = 1.0
        quality.append(outlier_ratio)
        return float(np.mean(quality))
