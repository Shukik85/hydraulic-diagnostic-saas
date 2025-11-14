"""
Production-grade database connection management with lifecycle, health checks,
and proper async pool handling.

Ключевые возможности:
- AsyncPG connection pool с proper lifecycle
- FastAPI dependency injection integration
- Connection health monitoring
- Automatic retry with exponential backoff
- Graceful shutdown
- Query timeout handling
- Connection pool metrics
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg
from loguru import logger

from config import get_settings

# ... (db manager implementation here) ... Код из файла 58 (database-manager.py)...