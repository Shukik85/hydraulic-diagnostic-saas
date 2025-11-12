# services/gnn_service/db_client.py
"""
TimescaleDB async client supporting multi-component and multi-system dynamic queries.
"""
import asyncpg
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from config import db_config

logger = logging.getLogger(__name__)

class TimescaleDBClient:
    def __init__(self, config=None):
        self.config = config or db_config
        self.pool = None
    async def connect(self):
        self.pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=2,
            max_size=self.config.pool_size,
            timeout=self.config.timeout
        )
        logger.info("TimescaleDB connection established")
    async def close(self):
        if self.pool:
            await self.pool.close()
            logger.info("TimescaleDB connection closed")
    async def query_sensor_data(self, system_id: str, start: datetime, end: datetime, timestep: int = 5) -> Dict[str, Any]:
        query = """
        SELECT time_bucket($1, timestamp) AS bucket, component_id, sensor_type, AVG(value) AS avg_val
        FROM sensor_readings
        WHERE system_id = $2 AND timestamp BETWEEN $3 AND $4
        GROUP BY bucket, component_id, sensor_type
        ORDER BY bucket ASC, component_id, sensor_type
        """
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, f"{timestep} minutes", system_id, start, end)
        result = {}
        for r in rows:
            comp = r["component_id"]
            if comp not in result:
                result[comp] = {}
            result[comp][r["sensor_type"]] = float(r["avg_val"]) if r["avg_val"] is not None else 0.0
        return result
    async def health_check(self):
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
# Singleton
_db_client = None
async def get_db_client():
    global _db_client
    if _db_client is None:
        _db_client = TimescaleDBClient()
        await _db_client.connect()
    return _db_client
