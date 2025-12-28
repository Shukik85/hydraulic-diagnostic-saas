"""Prepare ZeMA dataset into 10Hz cycle CSVs and optional edge_history CSVs.

Input:
  services/gnn_service/data/raw_real_dataset/*.txt
  - Each sensor file contains one cycle per line.
  - Each line contains the time-series samples for that cycle.

Output (default):
  services/gnn_service/data/processed/zema_10hz/cycles/cycle_00000.csv

Optional output (--edge-history):
  services/gnn_service/data/processed/zema_10hz/edge_history/{edge_id}/cycle_00000.csv

Resampling policy to 10Hz (600 samples for 60s):
- 100Hz signals (PS*, EPS1): decimate by factor 10 (take every 10th sample)
- 10Hz signals (FS1, FS2): keep as-is
- 1Hz signals (TS*, VS1, CE, CP, SE): repeat each sample 10 times

This script is streaming: it reads cycle lines one-by-one from all sensor files,
so it does not load entire dataset into RAM.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


RAW_DIR_DEFAULT = Path("services/gnn_service/data/raw_real_dataset")
OUT_DIR_DEFAULT = Path("services/gnn_service/data/processed/zema_10hz")
MAPPING_DEFAULT = Path("services/gnn_service/configs/zema_sensor_mapping.yaml")


SENSORS_100HZ = {"PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"}
SENSORS_10HZ = {"FS1", "FS2"}
SENSORS_1HZ = {"TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"}


@dataclass(frozen=True)
class EdgeMapping:
    edge_id: str
    sensors: dict  # sensor_name -> field meta


def _parse_floats_line(line: str) -> np.ndarray:
    """Parse a single cycle line into float array.

    Supports both tab-separated and whitespace-separated formats.
    """
    line = line.strip()
    if not line:
        return np.array([], dtype=np.float32)

    # Fast path: split by whitespace (also handles tabs)
    parts = line.split()
    return np.asarray(parts, dtype=np.float32)


def _to_10hz(sensor: str, arr: np.ndarray) -> np.ndarray:
    """Convert sensor array to 10Hz (600 samples)."""
    if arr.size == 0:
        return np.zeros(600, dtype=np.float32)

    if sensor in SENSORS_100HZ:
        # Expect 6000; decimate to 600
        return arr[::10][:600].astype(np.float32, copy=False)

    if sensor in SENSORS_10HZ:
        # Expect 600
        return arr[:600].astype(np.float32, copy=False)

    if sensor in SENSORS_1HZ:
        # Expect 60; repeat to 600
        repeated = np.repeat(arr[:60], 10)
        if repeated.size < 600:
            repeated = np.pad(repeated, (0, 600 - repeated.size), mode="edge")
        return repeated[:600].astype(np.float32, copy=False)

    # Unknown sensor: best-effort
    if arr.size >= 6000:
        return arr[::10][:600].astype(np.float32, copy=False)
    if arr.size == 600:
        return arr.astype(np.float32, copy=False)
    if arr.size == 60:
        return np.repeat(arr, 10)[:600].astype(np.float32, copy=False)

    # Fallback: interpolate to 600
    x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=600, endpoint=True)
    return np.interp(x_new, x_old, arr).astype(np.float32, copy=False)


def _read_profile(profile_path: Path) -> list[list[str]]:
    """Read profile.txt: one cycle per line."""
    rows: list[list[str]] = []
    with profile_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rows.append(parts)
    return rows


def _load_mapping(mapping_path: Path) -> list[EdgeMapping]:
    cfg = yaml.safe_load(mapping_path.read_text(encoding="utf-8"))
    edges = []
    for edge_id, ed in (cfg.get("edges") or {}).items():
        edges.append(EdgeMapping(edge_id=edge_id, sensors=ed.get("sensors") or {}))
    return edges


def _ensure_dirs(out_dir: Path, edge_history: bool, edge_mappings: list[EdgeMapping]) -> None:
    (out_dir / "cycles").mkdir(parents=True, exist_ok=True)
    if edge_history:
        base = out_dir / "edge_history"
        for em in edge_mappings:
            (base / em.edge_id).mkdir(parents=True, exist_ok=True)


def _iter_cycle_lines(sensor_path: Path) -> Iterable[np.ndarray]:
    with sensor_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield _parse_floats_line(line)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=str, default=str(RAW_DIR_DEFAULT))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    p.add_argument("--mapping", type=str, default=str(MAPPING_DEFAULT))
    p.add_argument("--limit-cycles", type=int, default=0, help="0 = all cycles")
    p.add_argument("--edge-history", action="store_true", help="Also write per-edge CSVs")
    args = p.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    mapping_path = Path(args.mapping)

    # Sensor files present in repo
    sensor_names = sorted(
        list(SENSORS_100HZ | SENSORS_10HZ | SENSORS_1HZ)
    )

    sensor_files = {s: raw_dir / f"{s}.txt" for s in sensor_names}
    missing = [s for s, fp in sensor_files.items() if not fp.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sensor files in {raw_dir}: {missing}")

    profile_path = raw_dir / "profile.txt"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile.txt in {raw_dir}")

    profile_rows = _read_profile(profile_path)

    edge_mappings = _load_mapping(mapping_path)
    _ensure_dirs(out_dir, args.edge_history, edge_mappings)

    # Open iterators for each sensor file
    iters = {s: _iter_cycle_lines(sensor_files[s]) for s in sensor_names}

    # Precompute time axis for 10Hz, 60 seconds => 600 samples
    t = np.arange(0, 60.0, 0.1, dtype=np.float32)  # 0..59.9

    n_cycles = 0
    while True:
        if args.limit_cycles and n_cycles >= args.limit_cycles:
            break

        # read one cycle from each sensor
        cycle_data: dict[str, np.ndarray] = {}
        try:
            for s in sensor_names:
                raw_arr = next(iters[s])
                cycle_data[s] = _to_10hz(s, raw_arr)
        except StopIteration:
            break

        # Prepare labels (best-effort: first 4 columns)
        labels = profile_rows[n_cycles] if n_cycles < len(profile_rows) else []
        labels4 = (labels + [""] * 4)[:4]

        cycle_id = f"cycle_{n_cycles:05d}"
        out_path = out_dir / "cycles" / f"{cycle_id}.csv"

        # Write combined cycle CSV
        header = ["t"] + sensor_names + [
            "cooler_condition",
            "valve_condition",
            "pump_condition",
            "accumulator_condition",
        ]

        with out_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(600):
                row = [float(t[i])] + [float(cycle_data[s][i]) for s in sensor_names] + labels4
                w.writerow(row)

        # Optional edge_history writing
        if args.edge_history:
            # Derived signals
            dp_filter = cycle_data["PS5"] - cycle_data["PS6"]

            for em in edge_mappings:
                edge_out = out_dir / "edge_history" / em.edge_id / f"{cycle_id}.csv"
                edge_sensor_cols = list(em.sensors.keys())

                # Include derived if needed
                derived_cols = []
                if em.edge_id == "zema_001_tank__zema_001_filter":
                    derived_cols = ["filter_dp"]

                edge_header = ["t"] + edge_sensor_cols + derived_cols

                with edge_out.open("w", newline="", encoding="utf-8") as ef:
                    ew = csv.writer(ef)
                    ew.writerow(edge_header)
                    for i in range(600):
                        vals = [float(t[i])]
                        for s in edge_sensor_cols:
                            vals.append(float(cycle_data[s][i]))
                        for d in derived_cols:
                            if d == "filter_dp":
                                vals.append(float(dp_filter[i]))
                        ew.writerow(vals)

        n_cycles += 1
        if n_cycles % 50 == 0:
            print(f"Processed {n_cycles} cycles")

    print(f"Done. Processed {n_cycles} cycles. Output: {out_dir}")


if __name__ == "__main__":
    main()
