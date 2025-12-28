"""Export a single ZeMA cycle into a unified time-aligned CSV.

Reads one row (cycle) from each sensor TXT file under data/raw_real_dataset/
(ZeMA hydraulic condition monitoring dataset format) and resamples all signals
onto a common time grid (default: 10 Hz).

Outputs a CSV suitable for quick plotting and for building edge_history
DataFrames later (GraphBuilderV2 expects columns like timestamp + sensor cols).

Usage:
  python services/gnn_service/scripts/zema_export_cycle_csv.py --cycle-id 0

Notes:
  - This script avoids loading whole TXT files into RAM; it streams until the
    requested line.
  - Default sensor sampling rates are based on the dataset conventions:
      PS1-PS6, EPS1: 100 Hz
      FS1-FS2: 10 Hz
      TS1-TS4, VS1, CE, CP, SE: 1 Hz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SENSOR_HZ: dict[str, int] = {
    # 100 Hz
    "PS1": 100,
    "PS2": 100,
    "PS3": 100,
    "PS4": 100,
    "PS5": 100,
    "PS6": 100,
    "EPS1": 100,
    # 10 Hz
    "FS1": 10,
    "FS2": 10,
    # 1 Hz
    "TS1": 1,
    "TS2": 1,
    "TS3": 1,
    "TS4": 1,
    "VS1": 1,
    "CE": 1,
    "CP": 1,
    "SE": 1,
}


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    hz: int


def _read_cycle_row(txt_path: Path, cycle_id: int) -> np.ndarray:
    """Read the N-th line (0-based) and parse floats."""
    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i == cycle_id:
                # ZeMA TXT lines are whitespace-separated
                arr = np.fromstring(line.strip(), sep=" ")
                if arr.size == 0:
                    # fallback: handle tabs/multiple spaces robustly
                    arr = np.array([float(x) for x in line.strip().split() if x], dtype=np.float64)
                return arr
    raise ValueError(f"cycle_id={cycle_id} out of range for {txt_path}")


def _resample_to_target(x: np.ndarray, orig_hz: int, target_hz: int) -> np.ndarray:
    if orig_hz == target_hz:
        return x.astype(np.float64, copy=False)

    if orig_hz > target_hz:
        ratio = orig_hz // target_hz
        if orig_hz % target_hz != 0:
            raise ValueError(f"Non-integer downsample ratio: {orig_hz} -> {target_hz}")
        n = (len(x) // ratio) * ratio
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        x = x[:n]
        return x.reshape(-1, ratio).mean(axis=1)

    # orig_hz < target_hz
    ratio = target_hz // orig_hz
    if target_hz % orig_hz != 0:
        raise ValueError(f"Non-integer upsample ratio: {orig_hz} -> {target_hz}")
    return np.repeat(x, ratio)


def _ensure_length(x: np.ndarray, target_len: int) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) > target_len:
        return x[:target_len]
    # pad with last value if exists, else zeros
    if len(x) == 0:
        return np.zeros(target_len, dtype=np.float64)
    pad = np.full(target_len - len(x), x[-1], dtype=np.float64)
    return np.concatenate([x, pad])


def export_cycle(
    data_dir: Path,
    cycle_id: int,
    sensors: list[SeriesSpec],
    target_hz: int,
    duration_s: int,
) -> pd.DataFrame:
    target_len = int(duration_s * target_hz)
    t = np.arange(target_len, dtype=np.float64) / float(target_hz)
    out: dict[str, np.ndarray] = {"timestamp_s": t}

    for spec in sensors:
        path = data_dir / f"{spec.name}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing sensor file: {path}")

        raw = _read_cycle_row(path, cycle_id=cycle_id)
        res = _resample_to_target(raw, orig_hz=spec.hz, target_hz=target_hz)
        res = _ensure_length(res, target_len)
        out[spec.name] = res

    return pd.DataFrame(out)


def _parse_sensors(arg: str | None) -> list[str]:
    if not arg:
        return list(DEFAULT_SENSOR_HZ.keys())
    return [s.strip() for s in arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle-id", type=int, required=True, help="0-based cycle index")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("services/gnn_service/data/raw_real_dataset"),
        help="Directory containing ZeMA raw TXT files",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: services/gnn_service/data/processed/zema_cycles/cycle_<id>_10hz.csv)",
    )
    parser.add_argument(
        "--sensors",
        type=str,
        default=None,
        help="Comma-separated sensor names (default: all known sensors)",
    )
    parser.add_argument("--target-hz", type=int, default=10, help="Resample frequency")
    parser.add_argument("--duration-s", type=int, default=60, help="Cycle duration in seconds")

    args = parser.parse_args()

    sensor_names = _parse_sensors(args.sensors)
    unknown = [s for s in sensor_names if s not in DEFAULT_SENSOR_HZ]
    if unknown:
        raise ValueError(f"Unknown sensors (add to DEFAULT_SENSOR_HZ): {unknown}")

    specs = [SeriesSpec(name=s, hz=DEFAULT_SENSOR_HZ[s]) for s in sensor_names]

    out_csv = args.out_csv
    if out_csv is None:
        out_csv = Path("services/gnn_service/data/processed/zema_cycles") / f"cycle_{args.cycle_id:05d}_{args.target_hz}hz.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = export_cycle(
        data_dir=args.data_dir,
        cycle_id=args.cycle_id,
        sensors=specs,
        target_hz=args.target_hz,
        duration_s=args.duration_s,
    )

    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote: {out_csv} (rows={len(df)}, cols={len(df.columns)})")


if __name__ == "__main__":
    main()
