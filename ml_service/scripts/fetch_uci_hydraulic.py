#!/usr/bin/env python3
"""
Fetch UCI Hydraulic System dataset and export tidy Parquet/CSV for ML service.
- Source: UCI #447 "Condition monitoring of hydraulic systems" (CC BY 4.0)
- Output:
  data/uci_hydraulic/
    - cycles.parquet (long format)
    - cycles_sample_100.parquet
    - cycles_sample_100.csv
    - README.md (schema)

Usage:
  python ml_service/scripts/fetch_uci_hydraulic.py --limit 100 --out data/uci_hydraulic
Requirements:
  pip install ucimlrepo pandas pyarrow numpy
"""
from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

SENSOR_MAP = {
    # Map UCI files to our logical sensor types with nominal Hz
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
    "EPS1": ("pressure", 1),  # virtual pressure metric
}

UNITS = {
    "pressure": "bar",
    "flow": "l/min",
    "temperature": "celsius",
    "vibration": "g",
}


def wide_row_to_long(df_row: pd.Series, sensor_name: str, sensor_type: str, hz: int, cycle_idx: int, system_id: str) -> pd.DataFrame:
    values = df_row.values.astype(float)
    n = len(values)
    # Build timestamps 0..(n-1)/hz seconds relative; assign an absolute synthetic ts index by cycle
    timestamps = np.arange(n) / float(hz)
    return pd.DataFrame(
        {
            "system_id": system_id,
            "cycle": cycle_idx,
            "timestamp": timestamps,
            "sensor": sensor_name,
            "sensor_type": sensor_type,
            "value": values,
            "unit": UNITS.get(sensor_type, ""),
            "component_id": None,
        }
    )


def build_long_format(sensors: dict, limit: int | None) -> pd.DataFrame:
    # Determine cycles count from one mandatory file, e.g., PS1
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
        # df rows are cycles; columns are samples within cycle
        for i in range(n_cycles):
            long_df = wide_row_to_long(df.iloc[i], name, stype, hz, i, system_id)
            frames.append(long_df)

    long = pd.concat(frames, ignore_index=True)
    # Ensure dtypes
    long["timestamp"] = long["timestamp"].astype("float32")
    long["value"] = long["value"].astype("float32")
    return long


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100, help="limit cycles for export")
    ap.add_argument("--out", type=str, default="data/uci_hydraulic", help="output dir")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch from UCI
    ds = fetch_ucirepo(id=447)
    # The dataset provides multiple data files; convert to dict of DataFrames
    sensors = {}
    for name in SENSOR_MAP.keys():
        try:
            # Access by name from ds.data.original (dict of DataFrames)
            sensors[name] = ds.data.original[name]
        except Exception:
            pass

    required = ["PS1", "FS1", "TS1", "VS1"]
    for r in required:
        if r not in sensors:
            raise RuntimeError(f"Required sensor file missing in dataset: {r}")

    long_df = build_long_format(sensors, args.limit)

    # Save Parquet and CSV sample
    parquet_path = out_dir / "cycles.parquet"
    sample_parquet_path = out_dir / "cycles_sample_100.parquet"
    sample_csv_path = out_dir / "cycles_sample_100.csv"

    long_df.to_parquet(parquet_path, index=False)

    sample = long_df.groupby(["system_id", "cycle", "sensor"]).head(100).reset_index(drop=True)
    sample.to_parquet(sample_parquet_path, index=False)
    sample.to_csv(sample_csv_path, index=False)

    # README with schema
    (out_dir / "README.md").write_text(
        """# UCI Hydraulic Export (Tidy)

Columns:
- system_id (uuid string)
- cycle (int) — UCI cycle index
- timestamp (float seconds within cycle)
- sensor (str: PS1..PS6, FS1..FS2, TS1..TS4, VS1, EPS1)
- sensor_type (str: pressure|flow|temperature|vibration)
- value (float)
- unit (str)
- component_id (nullable)

Notes:
- Generated from UCI #447 via ucimlrepo.
- One file per sensor originally (row = 60s cycle, columns = samples).
- Converted to long/tidy for ML service ingestion.
- Sample files include first 100 points per (system_id, cycle, sensor) group.
""",
        encoding="utf-8",
    )

    print(f"Saved: {parquet_path}")
    print(f"Saved: {sample_parquet_path}")
    print(f"Saved: {sample_csv_path}")


if __name__ == "__main__":
    main()
