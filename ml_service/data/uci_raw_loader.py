#!/usr/bin/env python3
"""
UCI Hydraulic RAW Loader
- Parse raw sensor .txt files from ml_service/data/raw/uci_hydraulic
- Validate, align, feature engineer, and export processed dataset
"""
from pathlib import Path
import pandas as pd
import numpy as np
import structlog
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = structlog.get_logger()

RAW_DIR = Path("ml_service/data/raw/uci_hydraulic")
PROCESSED_PARQUET = Path("ml_service/data/processed/uci_hydraulic.parquet")
EXPORT_CSV = Path("ml_service/data/industrial_iot/Industrial_fault_detection.csv")
REPORT_PATH = Path("ml_service/reports/raw_ingestion_report.json")

SENSOR_FILES = [
    "PS1.txt","PS2.txt","PS3.txt","PS4.txt","PS5.txt","PS6.txt",
    "FS1.txt","FS2.txt",
    "TS1.txt","TS2.txt","TS3.txt","TS4.txt",
    "VS1.txt","EPS1.txt"
]

COL_MAP = {
    "PS1.txt":"PS1","PS2.txt":"PS2","PS3.txt":"PS3","PS4.txt":"PS4","PS5.txt":"PS5","PS6.txt":"PS6",
    "FS1.txt":"FS1","FS2.txt":"FS2",
    "TS1.txt":"TS1","TS2.txt":"TS2","TS3.txt":"TS3","TS4.txt":"TS4",
    "VS1.txt":"VS1","EPS1.txt":"EPS1"
}


def _read_sensor(path: Path) -> pd.Series:
    s = pd.read_csv(path, sep="\t", header=None, squeeze=True, engine="python")
    return s.iloc[:,0] if isinstance(s, pd.DataFrame) else s


def load_raw_align() -> pd.DataFrame:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW directory not found: {RAW_DIR}")
    missing = [f for f in SENSOR_FILES if not (RAW_DIR/f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing sensor files: {missing}")

    series = {}
    with ThreadPoolExecutor(max_workers=min(8, len(SENSOR_FILES))) as ex:
        futs = {ex.submit(_read_sensor, RAW_DIR/f): f for f in SENSOR_FILES}
        for fut in as_completed(futs):
            fname = futs[fut]
            s = fut.result()
            series[COL_MAP[fname]] = s

    # align by length
    min_len = min(len(s) for s in series.values())
    for k in series:
        series[k] = series[k].iloc[:min_len].reset_index(drop=True)

    df = pd.DataFrame(series)
    # synthetic Timestamp @10 Hz
    ts = pd.date_range("2025-01-01", periods=min_len, freq="100L")
    df.insert(0, "Timestamp", ts)
    return df


def apply_profile_labels(df: pd.DataFrame) -> pd.DataFrame:
    profile_path = RAW_DIR/"profile.txt"
    if profile_path.exists():
        try:
            prof = pd.read_csv(profile_path, sep="\s+", header=None, engine="python")
            prof = prof.iloc[:len(df)].reset_index(drop=True)
            # assume columns: load, cooler, vib_flag, p_set, fault_flag
            vib_flag = prof.iloc[:,2] if prof.shape[1] >= 3 else 0
            fault_flag = prof.iloc[:,4] if prof.shape[1] >= 5 else 0
            fault_bin = ((fault_flag>0) | (vib_flag>0)).astype(int)
        except Exception as e:
            logger.warning("Failed to parse profile.txt, defaulting labels to 0", error=str(e))
            fault_bin = pd.Series(np.zeros(len(df), dtype=int))
    else:
        logger.warning("profile.txt not found, defaulting labels to 0")
        fault_bin = pd.Series(np.zeros(len(df), dtype=int))

    df["Fault Label"] = fault_bin.values
    return df


def engineer_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # example aggregates
    out["RMS_VS1_1s"] = out["VS1"].rolling(10, min_periods=1).apply(lambda x: np.sqrt(np.mean(x**2)))
    for col in ["PS1","PS2","PS3","PS4","PS5","PS6"]:
        out[f"{col}_std_5s"] = out[col].rolling(50, min_periods=1).std().fillna(0)
    return out


def export_processed(df: pd.DataFrame):
    PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    EXPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PARQUET, index=False)
    tidy = pd.DataFrame({
        "Timestamp": df["Timestamp"],
        "Vibration (mm/s)": df.get("VS1", pd.Series(np.nan, index=df.index)),
        "Temperature (°C)": df.get("TS1", pd.Series(np.nan, index=df.index)),
        "Pressure (bar)": df.get("PS1", pd.Series(np.nan, index=df.index)),
        "RMS Vibration": df.get("RMS_VS1_1s", pd.Series(np.nan, index=df.index)),
        "Mean Temp": df.get("TS1", pd.Series(np.nan, index=df.index)).rolling(50, min_periods=1).mean(),
        "Fault Label": df["Fault Label"].astype(int)
    })
    tidy.to_csv(EXPORT_CSV, index=False)


def build_raw_dataset() -> dict:
    report = {"status":"ok","errors":[],"rows":0}
    try:
        df = load_raw_align()
        df = apply_profile_labels(df)
        df = engineer_basic_features(df)
        export_processed(df)
        report["rows"] = int(len(df))
    except Exception as e:
        report["status"] = "failed"
        report["errors"].append(str(e))
        logger.error("RAW build failed", error=str(e))
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    r = build_raw_dataset()
    print(json.dumps(r, indent=2))
