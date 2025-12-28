"""Apply sensor-to-topology mapping (JSON) to a cycle dataframe.

Input df_cycle can be:
- pandas.DataFrame (recommended for now)
- polars.DataFrame (supported)

Output:
- dict[edge_id, DataFrame] where DataFrame includes columns:
  - t_idx, t_s (if present)
  - sensor columns belonging to that edge

This keeps the user-facing dataset as a single Parquet file,
while still enabling edge-centric features internally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_mapping_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def apply_mapping_to_edges(df_cycle: Any, mapping: dict[str, Any]) -> dict[str, Any]:
    edges_cfg = (mapping.get("edges") or {})

    # Detect backend
    is_polars = df_cycle.__class__.__module__.startswith("polars")

    out: dict[str, Any] = {}
    for edge_id, edge_def in edges_cfg.items():
        sensors = list((edge_def.get("sensors") or {}).keys())
        base_cols = []
        for c in ("t_idx", "t_s"):
            if _has_col(df_cycle, c, is_polars):
                base_cols.append(c)

        cols = base_cols + [s for s in sensors if _has_col(df_cycle, s, is_polars)]

        if is_polars:
            out[edge_id] = df_cycle.select(cols)
        else:
            out[edge_id] = df_cycle[cols].copy()

    return out


def _has_col(df: Any, col: str, is_polars: bool) -> bool:
    if is_polars:
        return col in df.columns
    return col in getattr(df, "columns", [])
