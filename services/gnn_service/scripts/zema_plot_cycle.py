"""Plot ZeMA cycle signals from a CSV created by zema_export_cycle_csv.py.

Creates:
  - One multi-panel PNG with all sensors stacked.
  - Optionally individual PNG per sensor.

Usage:
  python services/gnn_service/scripts/zema_plot_cycle.py \
    --csv services/gnn_service/data/processed/zema_cycles/cycle_00000_10hz.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_stacked(df: pd.DataFrame, out_path: Path, max_cols: int = 1) -> None:
    cols = [c for c in df.columns if c != "timestamp_s"]
    if not cols:
        raise ValueError("No sensor columns found in CSV")

    n = len(cols)
    fig_h = max(4.0, 1.6 * n)

    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(14, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    t = df["timestamp_s"].to_numpy()

    for ax, col in zip(axes, cols, strict=False):
        ax.plot(t, df[col].to_numpy(), linewidth=1.0)
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("t, s")
    fig.suptitle(out_path.stem, y=0.995)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_individual(df: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in df.columns if c != "timestamp_s"]
    t = df["timestamp_s"].to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)

    for col in cols:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(t, df[col].to_numpy(), linewidth=1.0)
        ax.set_title(col)
        ax.set_xlabel("t, s")
        ax.set_ylabel("value")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{col}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV path")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <csv_dir>/plots/<csv_stem>/)",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="Also export individual plots per sensor",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "timestamp_s" not in df.columns:
        raise ValueError("CSV must contain timestamp_s column")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = args.csv.parent / "plots" / args.csv.stem

    stacked_path = out_dir / f"{args.csv.stem}_stacked.png"
    plot_stacked(df, stacked_path)
    print(f"✅ Wrote: {stacked_path}")

    if args.individual:
        plot_individual(df, out_dir / "individual")
        print(f"✅ Wrote individual plots to: {out_dir / 'individual'}")


if __name__ == "__main__":
    main()
