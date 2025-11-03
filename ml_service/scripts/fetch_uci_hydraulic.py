#!/usr/bin/env python3
"""
Fetch UCI Hydraulic System dataset (direct HTTP) and export tidy Parquet/CSV for ML service.
- Source: UCI #447 (Condition monitoring of hydraulic systems)
- Fallback: use local raw files if download is unavailable

Usage:
  python ml_service/scripts/fetch_uci_hydraulic.py --limit 100 --out data/uci_hydraulic

Notes:
- If local raw files exist in data/raw/uci_hydraulic/*.txt, uses them.
- Otherwise tries to download from UCI; if unavailable, prompts to manual download.
"""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import requests

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/"
FILES = [
    "PS1.txt",
    "PS2.txt",
    "PS3.txt",
    "PS4.txt",
    "PS5.txt",
    "PS6.txt",
    "FS1.txt",
    "FS2.txt",
    "TS1.txt",
    "TS2.txt",
    "TS3.txt",
    "TS4.txt",
    "VS1.txt",
    "EPS1.txt",
    "profile.txt",
]

SENSOR_MAP = {
    "PS1": ("pressure", 100),
    "PS2": ("pressure", 100),
    "PS3": ("pressure", 100),
    "PS4": ("pressure", 100),
    "PS5": ("pressure", 100),
    "PS6": ("pressure", 100),
    "FS1": ("flow", 10),
    "FS2": ("flow", 10),
    "TS1": ("temperature", 1),
    "TS2": ("temperature", 1),
    "TS3": ("temperature", 1),
    "TS4": ("temperature", 1),
    "VS1": ("vibration", 1),
    "EPS1": ("pressure", 1),
}

UNITS = {"pressure": "bar", "flow": "l/min", "temperature": "celsius", "vibration": "g"}


def http_download(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, timeout=60, stream=True) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return True
    except Exception:
        return False


def ensure_raw_files(raw_dir: Path) -> Dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for fn in FILES:
        p = raw_dir / fn
        if not p.exists():
            url = f"{UCI_BASE}{fn}"
            ok = http_download(url, p)
            if not ok:
                print(f"WARN: Could not download {fn} from UCI. Please place it at {p}")
        if p.exists():
            paths[fn] = p
    # Minimal required for quick start
    for req in ("PS1.txt", "FS1.txt", "TS1.txt", "VS1.txt"):
        if req not in paths:
            raise RuntimeError(f"Missing required file: {req}. Download manually to {raw_dir}")
    return paths


def load_sensor_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\s+", header=None, engine="python")


def wide_row_to_long(
    row: pd.Series, sensor_name: str, sensor_type: str, hz: int, cycle: int, system_id: str
) -> pd.DataFrame:
    values = row.values.astype(float)
    n = len(values)
    timestamps = np.arange(n) / float(hz)
    return pd.DataFrame(
        {
            "system_id": system_id,
            "cycle": cycle,
            "timestamp": timestamps.astype("float32"),
            "sensor": sensor_name,
            "sensor_type": sensor_type,
            "value": values.astype("float32"),
            "unit": UNITS.get(sensor_type, ""),
            "component_id": None,
        }
    )


def build_long(sensors: Dict[str, pd.DataFrame], limit: int | None) -> pd.DataFrame:
    base = sensors["PS1"]
    n_cycles = base.shape[0]
    if limit:
        n_cycles = min(n_cycles, limit)
    system_id = str(uuid.uuid4())

    frames: List[pd.DataFrame] = []
    for name, (stype, hz) in SENSOR_MAP.items():
        if name not in sensors:
            continue
        df = sensors[name].iloc[:n_cycles]
        for i in range(n_cycles):
            frames.append(wide_row_to_long(df.iloc[i], name, stype, hz, i, system_id))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", type=str, default="data/uci_hydraulic")
    ap.add_argument("--raw", type=str, default="data/raw/uci_hydraulic")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)

    paths = ensure_raw_files(raw_dir)

    sensors: Dict[str, pd.DataFrame] = {}
    for key in SENSOR_MAP.keys():
        fn = f"{key}.txt"
        if fn in paths:
            sensors[key] = load_sensor_matrix(paths[fn])

    long_df = build_long(sensors, args.limit)

    parquet_path = out_dir / "cycles.parquet"
    sample_parquet = out_dir / "cycles_sample_100.parquet"
    sample_csv = out_dir / "cycles_sample_100.csv"

    long_df.to_parquet(parquet_path, index=False)
    sample = long_df.groupby(["system_id", "cycle", "sensor"]).head(100).reset_index(drop=True)
    sample.to_parquet(sample_parquet, index=False)
    sample.to_csv(sample_csv, index=False)

    (out_dir / "README.md").write_text(
        """# UCI Hydraulic Export (Tidy)

Columns:
- system_id (uuid), cycle (int), timestamp (float seconds),
- sensor (PS1..), sensor_type (pressure|flow|temperature|vibration),
- value (float), unit (str), component_id (nullable)

Source: UCI #447 direct files from machine-learning-databases/00360.
If files were missing, script printed warnings — download manually into data/raw/uci_hydraulic.
""",
        encoding="utf-8",
    )

    print(f"Saved: {parquet_path}")
    print(f"Saved: {sample_parquet}")
    print(f"Saved: {sample_csv}")


if __name__ == "__main__":
    main()
