"""
Feature Engineering for Hydraulic Systems Diagnostics
Enterprise feature extraction with resampling & gap handling
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import structlog
from scipy import stats

from api.schemas import FeatureVector, SensorDataBatch
from config import settings

logger = structlog.get_logger()


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
    "pressure": {"bar": 1.0, "psi": 0.0689476, "kpa": 0.01, "mpa": 10.0},
    "temperature": {"c": lambda x: x, "f": lambda x: (x - 32) * 5.0 / 9.0, "k": lambda x: x - 273.15},
    "flow": {"l/min": 1.0, "lpm": 1.0, "m3/h": 16.6667},
    "vibration": {"mm/s": 1.0},
}

INDUSTRIAL_DEFAULTS = {"temperature_mean": 40.0, "flow_mean": 10.0, "vibration_rms": 0.5}


class FeatureEngineer:
    def __init__(self):
        self.feature_cache = {}
        self.sampling_frequency = settings.sampling_frequency_hz
        self.window_size = int(settings.feature_window_minutes * 60 * self.sampling_frequency)
        # Resample params
        self.base_hz = getattr(settings, "feature_base_hz", 20)
        self.gap_small_ms = getattr(settings, "gap_small_ms", 250)
        self.gap_medium_s = getattr(settings, "gap_medium_s", 3)

    def _canonical_type(self, raw_type: str) -> str:
        rt = (raw_type or "").lower()
        for canonical, aliases in SENSOR_TYPE_MAPPING.items():
            if rt in aliases:
                return canonical
        if rt.startswith("ps"):
            return "pressure"
        if rt.startswith("ts"):
            return "temperature"
        if rt.startswith("fs"):
            return "flow"
        if rt.startswith("vs"):
            return "vibration"
        return rt

    def _convert_unit(self, value: float, unit: str | None, canonical: str) -> float:
        if unit is None:
            return float(value)
        u = unit.lower()
        conv = UNIT_CONVERSIONS.get(canonical, {})
        if u in conv:
            factor = conv[u]
            return float(factor(value) if callable(factor) else value * factor)
        return float(value)

    def _estimate_fs(self, ts: pd.Series) -> float:
        if len(ts) < 3:
            return float(self.base_hz)
        diffs = ts.sort_values().diff().dropna().dt.total_seconds().values
        if len(diffs) == 0:
            return float(self.base_hz)
        med = np.median(diffs)
        return float(1.0 / med) if med > 0 else float(self.base_hz)

    def _resample_series(self, df: pd.DataFrame, sensor_type: str) -> tuple[pd.Series, float, float]:
        s = df[df["sensor_type"] == sensor_type][["timestamp", "value"]].copy()
        if s.empty:
            return pd.Series(dtype=float), 0.0, 0.0
        s = s.set_index("timestamp").sort_index()
        fs_est = self._estimate_fs(s.index.to_series())
        # Target grid
        rule_ms = int(1000 / self.base_hz)
        grid = pd.date_range(start=s.index.min(), end=s.index.max(), freq=f"{rule_ms}ms")
        # Down/Up sample
        if fs_est >= self.base_hz:
            # downsample: бинируем по grid с mean
            res = s.resample(f"{rule_ms}ms").mean().reindex(grid)
        else:
            # upsample: интерполяция
            res = s.reindex(grid).interpolate(method="time")
        # Gap handling
        diffs_ms = res.index.to_series().diff().dt.total_seconds().fillna(0) * 1000
        # Простая метрика покрытия
        coverage = float(res["value"].notna().sum() / max(len(res), 1))
        # Малые дыры — интерполяция уже сделана; средние — ffill/bfill лимитированно
        res_ff = res["value"].ffill(limit=int(self.gap_medium_s * self.base_hz))
        res_fb = res_ff.bfill(limit=int(self.gap_medium_s * self.base_hz))
        return res_fb, fs_est, coverage

    def _build_resampled_table(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        # Определяем общий диапазон времени по всем сенсорам
        tmin = df["timestamp"].min()
        tmax = df["timestamp"].max()
        if pd.isna(tmin) or pd.isna(tmax) or tmin == tmax:
            return pd.DataFrame(), {}
        rule_ms = int(1000 / self.base_hz)
        grid = pd.date_range(start=tmin, end=tmax, freq=f"{rule_ms}ms")
        table = pd.DataFrame(index=grid)
        coverages: dict[str, float] = {}
        for st in [
            "pressure",
            "temperature",
            "flow",
            "vibration",
            "motor_power",
            "cooling_efficiency",
            "cooling_power",
            "system_efficiency",
        ]:
            series, fs_est, cov = self._resample_series(df, st)
            table[st] = series.reindex(grid)
            coverages[st] = cov
        return table, coverages

    async def extract_features(
        self, sensor_data: SensorDataBatch, feature_groups: list[str] | None = None
    ) -> FeatureVector:
        start_time = time.time()
        if feature_groups is None:
            feature_groups = ["sensor_features", "derived_features", "window_features"]
        # Canonicalize & unit convert
        data = []
        for r in sensor_data.readings:
            ctype = self._canonical_type(r.sensor_type)
            val = self._convert_unit(r.value, getattr(r, "unit", None), ctype)
            data.append(
                {
                    "timestamp": pd.to_datetime(r.timestamp),
                    "sensor_type": ctype,
                    "value": val,
                    "unit": getattr(r, "unit", None),
                    "component_id": r.component_id,
                }
            )
        raw = pd.DataFrame(data).sort_values("timestamp")
        logger.info(
            "FeatureEngineer input",
            system_id=str(sensor_data.system_id),
            sensor_types=list(map(str, raw["sensor_type"].unique())),
            total=len(raw),
        )
        # Resample to base grid
        resampled, coverages = self._build_resampled_table(raw)
        features: dict[str, float] = {}
        feature_names: list[str] = []

        def safe_stats(series: pd.Series, name_prefix: str) -> None:
            ser = series.dropna()
            if len(ser) > 0:
                features[f"{name_prefix}_mean"] = float(ser.mean())
                features[f"{name_prefix}_std"] = float(ser.std())
                features[f"{name_prefix}_max"] = float(ser.max())
                features[f"{name_prefix}_min"] = float(ser.min())
            else:
                defaults = {
                    f"{name_prefix}_mean": INDUSTRIAL_DEFAULTS.get(f"{name_prefix}_mean", 0.0),
                    f"{name_prefix}_std": 0.0,
                    f"{name_prefix}_max": 0.0,
                    f"{name_prefix}_min": 0.0,
                }
                features.update(defaults)

        # Sensor features
        if "sensor_features" in feature_groups and not resampled.empty:
            safe_stats(resampled.get("pressure", pd.Series(dtype=float)), "pressure")
            safe_stats(resampled.get("temperature", pd.Series(dtype=float)), "temperature")
            safe_stats(resampled.get("flow", pd.Series(dtype=float)), "flow")
            vib = resampled.get("vibration", pd.Series(dtype=float)).dropna()
            features["vibration_rms"] = (
                float(np.sqrt(np.mean(vib.values**2)))
                if len(vib) > 0
                else INDUSTRIAL_DEFAULTS.get("vibration_rms", 0.0)
            )
            features["system_efficiency"] = float(
                resampled.get("system_efficiency", pd.Series(dtype=float)).dropna().mean() or 0.0
            )
            features["cooling_efficiency"] = float(
                resampled.get("cooling_efficiency", pd.Series(dtype=float)).dropna().mean() or 0.0
            )
            features["cooling_power"] = float(
                resampled.get("cooling_power", pd.Series(dtype=float)).dropna().mean() or 0.0
            )
            features["motor_power_mean"] = float(
                resampled.get("motor_power", pd.Series(dtype=float)).dropna().mean() or 0.0
            )

        # Derived
        if "derived_features" in feature_groups and not resampled.empty:

            def gradient(ser: pd.Series) -> float:
                ser = ser.dropna()
                if len(ser) > 1:
                    return float(np.mean(np.gradient(ser.values)))
                return 0.0

            p = resampled.get("pressure", pd.Series(dtype=float))
            t = resampled.get("temperature", pd.Series(dtype=float))
            f = resampled.get("flow", pd.Series(dtype=float))
            features["pressure_gradient"] = gradient(p)
            features["temperature_gradient"] = gradient(t)
            features["flow_gradient"] = gradient(f)
            # Корреляции на общем отрезке
            min_len = min(p.dropna().shape[0], t.dropna().shape[0])
            if min_len >= int(getattr(settings, "min_overlap_seconds_for_corr", 5) * self.base_hz):
                p_v = p.dropna().values[:min_len]
                t_v = t.dropna().values[:min_len]
                corr = np.corrcoef(p_v, t_v)[0, 1]
                features["temp_pressure_correlation"] = float(0.0 if np.isnan(corr) else corr)
            else:
                features["temp_pressure_correlation"] = 0.0
            p_mean = float(p.dropna().mean()) if p.dropna().size > 0 else 0.0
            f_mean = float(f.dropna().mean()) if f.dropna().size > 0 else 0.0
            features["pressure_flow_ratio"] = float(p_mean / max(f_mean, 1e-3))

        # Window features
        if "window_features" in feature_groups and not resampled.empty:
            p = resampled.get("pressure", pd.Series(dtype=float)).dropna()
            if len(p) > 2:
                idx = np.arange(len(p))
                slope, _, _, _, _ = stats.linregress(idx, p.values)
                features["trend_slope"] = float(slope)
            else:
                features["trend_slope"] = 0.0
            if len(p) > 10:
                arr = p.values
                autocorr = np.corrcoef(arr[:-1], arr[1:])[0, 1]
                features["autocorrelation_lag1"] = float(0.0 if np.isnan(autocorr) else autocorr)
            else:
                features["autocorrelation_lag1"] = 0.0
            t = resampled.get("temperature", pd.Series(dtype=float)).dropna()
            if len(t) > 20:
                fft = np.fft.fft(t.values)
                power = np.abs(fft) ** 2
                dom = float(np.max(power[1:])) if len(power) > 1 else 0.0
                tot = float(np.sum(power[1:])) if len(power) > 1 else 1.0
                features["seasonality_score"] = float(min(1.0, dom / max(tot, 1e-6)))
            else:
                features["seasonality_score"] = 0.0

        # Coverage → влияет на уверенность (можно учесть в Ensemble позже)
        for k, cov in coverages.items():
            features[f"coverage_{k}"] = float(cov)
        cov_keys = [k for k in ["pressure", "temperature", "flow", "vibration"] if k in coverages]
        if cov_keys:
            features["global_coverage"] = float(np.mean([coverages[k] for k in cov_keys]))
        else:
            features["global_coverage"] = 0.0

        extraction_time = (time.time() - start_time) * 1000
        logger.debug(
            "Feature extraction completed",
            features_count=len(features),
            extraction_time_ms=extraction_time,
            data_quality=features.get("global_coverage", 0.0),
        )

        return FeatureVector(
            features=features,
            feature_names=list(features.keys()),
            extraction_time_ms=extraction_time,
            data_quality_score=float(features.get("global_coverage", 0.0)),
        )
