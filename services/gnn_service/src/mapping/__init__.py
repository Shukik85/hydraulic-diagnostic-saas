"""Mapping utilities.

- Load JSON mapping (zema_sensor_mapping.json)
- Apply mapping: cycle dataframe -> per-edge dataframe

This is the foundation for:
- building EdgeSensorReading/edge_history for GraphBuilderV2
- later: converting user-provided CSV/Parquet + mapping into inference requests
"""

from .apply_mapping import apply_mapping_to_edges, load_mapping_json

__all__ = [
    "apply_mapping_to_edges",
    "load_mapping_json",
]
