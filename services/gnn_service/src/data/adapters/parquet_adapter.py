"""Parquet adapter for offline datasets.

This adapter provides a minimal, production-friendly interface for reading a single
Parquet file produced by prepare_zema_10hz_dataset.py.

Expected schema (row-wise time series):
- cycle_id, t_idx, t_s
- sensor columns
- optional label columns

Notes:
- Uses polars if available (fast), otherwise falls back to pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ParquetDatasetConfig:
    path: Path


class ParquetAdapter:
    def __init__(self, config: ParquetDatasetConfig):
        self.config = config

    def iter_cycles(self) -> Iterator[tuple[int, "DataFrameLike"]]:
        """Yield (cycle_id, df_cycle) for each cycle."""
        path = self.config.path

        try:
            import polars as pl

            df = pl.read_parquet(path)
            for cid in df.select(pl.col("cycle_id").unique()).to_series().to_list():
                yield int(cid), df.filter(pl.col("cycle_id") == cid)
            return
        except Exception:
            pass

        import pandas as pd

        df = pd.read_parquet(path)
        for cid, g in df.groupby("cycle_id", sort=True):
            yield int(cid), g
