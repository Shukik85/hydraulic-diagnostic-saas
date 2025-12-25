"""Data adapters for sensor readings from multiple sources.

Supports:
    - TimescaleDB (mock and real)
    - CSV files
    - REST API (Phase 2)
    - Modbus TCP/RTU (Phase 2)
    - OPC-UA (Phase 2)

Examples:
    >>> from src.data.adapters import create_timescaledb_adapter
    >>> adapter = create_timescaledb_adapter()  # Auto mock/real based on .env
"""

from __future__ import annotations

from src.data.adapters.timescaledb_mock import (
    TimescaleDBMockAdapter,
    TimescaleDBRealAdapter,
    create_timescaledb_adapter,
)

__all__ = [
    "TimescaleDBMockAdapter",
    "TimescaleDBRealAdapter",
    "create_timescaledb_adapter",
]
