"""Prepare ZeMA raw dataset into a single Parquet file (10Hz canonical format).

Why:
- The previous per-cycle CSV export creates thousands of files, which is user-unfriendly.
- A single Parquet is easier to upload, version, and ingest.

Input:
  services/gnn_service/data/raw_real_dataset/*.txt
  - Each sensor file contains one cycle per line.
  - Each line contains the time-series samples for that cycle.

Output (default):
  services/gnn_service/data/processed/zema_10hz/zema_10hz.parquet

Resampling policy to 10Hz (600 samples for 60s):
- 100Hz signals (PS*, EPS1): decimate by factor 10 (take every 10th sample)
- 10Hz signals (FS1, FS2): keep as-is
- 1Hz signals (TS*, VS1, CE, CP, SE): repeat each sample 10 times

Parquet schema (row = one timestamp sample):
- cycle_id: int32
- t_idx: int32 (0..599)
- t_s: float32 (seconds)
- sensor columns (float32)
- labels: cooler_condition, valve_condition, pump_condition, accumulator_condition (string)

Notes:
- This script is streaming: it reads cycle lines one-by-one from all sensor files.
- Requires: pyarrow
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

RAW_DIR_DEFAULT = Path("services/gnn_service/data/raw_real_dataset")
OUT_DIR_DEFAULT = Path("services/gnn_service/data/processed/zema_10hz")

SENSORS_100HZ = {"PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "EPS1"}
SENSORS_10HZ = {"FS1", "FS2"}
SENSORS_1HZ = {"TS1", "TS2", "TS3", "TS4", "VS1", "CE", "CP", "SE"}


def _parse_floats_line(line: str) -> np.ndarray:
    line = line.strip()
    if not line:
        return np.array([], dtype=np.float32)
    return np.asarray(line.split(), dtype=np.float32)


def _to_10hz(sensor: str, arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros(600, dtype=np.float32)

    if sensor in SENSORS_100HZ:
        return arr[::10][:600].astype(np.float32, copy=False)

    if sensor in SENSORS_10HZ:
        return arr[:600].astype(np.float32, copy=False)

    if sensor in SENSORS_1HZ:
        repeated = np.repeat(arr[:60], 10)
        if repeated.size < 600:
            repeated = np.pad(repeated, (0, 600 - repeated.size), mode="edge")
        return repeated[:600].astype(np.float32, copy=False)

    if arr.size >= 6000:
        return arr[::10][:600].astype(np.float32, copy=False)
    if arr.size == 600:
        return arr.astype(np.float32, copy=False)
    if arr.size == 60:
        return np.repeat(arr, 10)[:600].astype(np.float32, copy=False)

    x_old = np.linspace(0.0, 1.0, num=arr.size, endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=600, endpoint=True)
    return np.interp(x_new, x_old, arr).astype(np.float32, copy=False)


def _iter_cycle_lines(sensor_path: Path) -> Iterable[np.ndarray]:
    with sensor_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            yield _parse_floats_line(line)


def _read_profile_rows(profile_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with profile_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                rows.append(parts)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=str, default=str(RAW_DIR_DEFAULT))
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR_DEFAULT))
    p.add_argument("--out-file", type=str, default="")
    p.add_argument("--limit-cycles", type=int, default=0, help="0 = all cycles")
    p.add_argument("--compression", type=str, default="zstd", help="Parquet compression")
    args = p.parse_args()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for Parquet export. Install: pip install pyarrow"
        ) from e

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = Path(args.out_file) if args.out_file else (out_dir / "zema_10hz.parquet")

    sensor_names = sorted(list(SENSORS_100HZ | SENSORS_10HZ | SENSORS_1HZ))
    sensor_files = {s: raw_dir / f"{s}.txt" for s in sensor_names}

    missing = [s for s, fp in sensor_files.items() if not fp.exists()]
    if missing:
        raise FileNotFoundError(f"Missing sensor files in {raw_dir}: {missing}")

    profile_path = raw_dir / "profile.txt"
    if not profile_path.exists():
        raise FileNotFoundError(f"Missing profile.txt in {raw_dir}")

    profile_rows = _read_profile_rows(profile_path)

    iters = {s: _iter_cycle_lines(sensor_files[s]) for s in sensor_names}

    t_s = np.arange(0, 60.0, 0.1, dtype=np.float32)  # 600 samples
    t_idx = np.arange(0, 600, dtype=np.int32)

    writer: pq.ParquetWriter | None = None

    n_cycles = 0
    while True:
        if args.limit_cycles and n_cycles >= args.limit_cycles:
            break

        cycle_data: dict[str, np.ndarray] = {}
        try:
            for s in sensor_names:
                raw_arr = next(iters[s])
                cycle_data[s] = _to_10hz(s, raw_arr)
        except StopIteration:
            break

        labels = profile_rows[n_cycles] if n_cycles < len(profile_rows) else []
        labels4 = (labels + [""] * 4)[:4]

        batch_dict: dict[str, object] = {
            "cycle_id": np.full(600, n_cycles, dtype=np.int32),
            "t_idx": t_idx,
            "t_s": t_s,
        }
        for s in sensor_names:
            batch_dict[s] = cycle_data[s].astype(np.float32, copy=False)

        batch_dict["cooler_condition"] = np.full(600, labels4[0], dtype=object)
        batch_dict["valve_condition"] = np.full(600, labels4[1], dtype=object)
        batch_dict["pump_condition"] = np.full(600, labels4[2], dtype=object)
        batch_dict["accumulator_condition"] = np.full(600, labels4[3], dtype=object)

        table = pa.Table.from_pydict(batch_dict)

        if writer is None:
            writer = pq.ParquetWriter(
                where=str(out_file),
                schema=table.schema,
                compression=args.compression,
                use_dictionary=True,
            )

        writer.write_table(table)

        n_cycles += 1
        if n_cycles % 50 == 0:
            print(f"Processed {n_cycles} cycles")

    if writer is not None:
        writer.close()

    print(f"Done. Processed {n_cycles} cycles. Output: {out_file}")


if __name__ == "__main__":
    main()
