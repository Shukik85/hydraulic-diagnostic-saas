"""TimescaleDB connector для sensor data.

Async PostgreSQL client для:
- Sensor time-series queries
- Equipment metadata retrieval
- Batch fetching
- Connection pooling

Python 3.14 Features:
    - Deferred annotations
    - Union types
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

try:
    import asyncpg
except ImportError:
    # asyncpg will be available at runtime
    asyncpg = None  # type: ignore[assignment]

import pandas as pd

if TYPE_CHECKING:

    from src.schemas import TimeWindow

logger = logging.getLogger(__name__)


class TimescaleConnector:
    """Async connector для TimescaleDB.

    Использует asyncpg для high-performance async queries.

    Features:
        - Connection pooling (2-10 connections)
        - Automatic retry с exponential backoff
        - Query timeout handling
        - Batch fetching optimization

    Args:
        db_url: PostgreSQL connection URL
        pool_min_size: Minimum pool size
        pool_max_size: Maximum pool size
        command_timeout: Query timeout в секундах
        max_retries: Maximum retry attempts

    Examples:
        >>> connector = TimescaleConnector(db_url=DATABASE_URL)
        >>> await connector.connect()
        >>>
        >>> data = await connector.fetch_sensor_data(
        ...     equipment_id="excavator_001",
        ...     time_window=TimeWindow(start_time=..., end_time=...),
        ...     sensors=["pressure_pump_out", "temperature_fluid"]
        ... )
        >>>
        >>> await connector.close()
    """

    def __init__(
        self,
        db_url: str,
        pool_min_size: int = 2,
        pool_max_size: int = 10,
        command_timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.db_url = db_url
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self.command_timeout = command_timeout
        self.max_retries = max_retries

        self.pool: Any = None  # asyncpg.Pool or None
        self._connected = False

    async def connect(self) -> None:
        """Установить connection pool.

        Raises:
            ConnectionError: Если не удалось подключиться
        """
        if asyncpg is None:
            msg = "asyncpg not installed. Install with: pip install asyncpg"
            raise ImportError(msg)

        try:
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=self.pool_min_size,
                max_size=self.pool_max_size,
                command_timeout=self.command_timeout,
            )
            self._connected = True
            logger.info(
                f"Connected to TimescaleDB (pool: {self.pool_min_size}-{self.pool_max_size})"
            )
        except Exception as e:
            logger.exception(f"Failed to connect to TimescaleDB: {e}")
            msg = f"TimescaleDB connection failed: {e}"
            raise ConnectionError(msg) from e

    async def close(self) -> None:
        """Закрыть connection pool."""
        if self.pool is not None:
            await self.pool.close()
            self._connected = False
            logger.info("TimescaleDB connection closed")

    @asynccontextmanager
    async def get_connection(self) -> Any:
        """Получить connection из pool (context manager).

        Yields:
            conn: asyncpg.Connection

        Raises:
            RuntimeError: Если pool не инициализирован
        """
        if self.pool is None:
            msg = "Connection pool not initialized. Call connect() first."
            raise RuntimeError(msg)

        async with self.pool.acquire() as conn:
            yield conn

    async def _execute_with_retry(
        self, query_func: Any, *args: Any, **kwargs: Any
    ) -> Any:
        """Выполнить query с retry logic.

        Args:
            query_func: Async function для выполнения
            *args: Positional arguments для query_func
            **kwargs: Keyword arguments для query_func

        Returns:
            result: Query result

        Raises:
            Exception: После исчерпания retries
        """
        for attempt in range(self.max_retries):
            try:
                result = await query_func(*args, **kwargs)
                return result
            except (TimeoutError, Exception) as e:  # asyncpg.PostgresError if available
                if attempt == self.max_retries - 1:
                    logger.exception(f"Query failed after {self.max_retries} attempts: {e}")
                    raise

                # Exponential backoff
                wait_time = 2**attempt
                logger.warning(
                    f"Query failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)

        msg = "Should not reach here"
        raise RuntimeError(msg)

    async def fetch_sensor_data(
        self, equipment_id: str, time_window: TimeWindow, sensors: list[str]
    ) -> pd.DataFrame:
        """Получить sensor data из TimescaleDB.

        Args:
            equipment_id: Equipment identifier
            time_window: Time range для query
            sensors: Список sensor names

        Returns:
            df: DataFrame с sensor readings [timestamp, sensor1, sensor2, ...]

        Raises:
            ValueError: Если sensors list пуст
            RuntimeError: Если query failed

        Examples:
            >>> data = await connector.fetch_sensor_data(
            ...     equipment_id="exc_001",
            ...     time_window=TimeWindow(
            ...         start_time=datetime(2025, 11, 1),
            ...         end_time=datetime(2025, 11, 21)
            ...     ),
            ...     sensors=["pressure_pump", "temperature"]
            ... )
            >>> data.shape  # (N_samples, 3) - timestamp + 2 sensors
        """
        if not sensors:
            msg = "Sensors list cannot be empty"
            raise ValueError(msg)

        async def _fetch() -> pd.DataFrame:
            query = """
                SELECT
                    timestamp,
                    {sensor_columns}
                FROM sensor_data
                WHERE equipment_id = $1
                  AND timestamp >= $2
                  AND timestamp <= $3
                ORDER BY timestamp ASC
            """.format(sensor_columns=", ".join(sensors))

            async with self.get_connection() as conn:
                rows = await conn.fetch(
                    query, equipment_id, time_window.start_time, time_window.end_time
                )

            # Convert to DataFrame
            if not rows:
                logger.warning(
                    f"No data found for {equipment_id} in time window {time_window}"
                )
                return pd.DataFrame(columns=["timestamp", *sensors])

            df = pd.DataFrame(rows, columns=["timestamp", *sensors])
            logger.info(f"Fetched {len(df)} samples for {equipment_id}")

            return df

        result = await self._execute_with_retry(_fetch)
        if not isinstance(result, pd.DataFrame):
            msg = f"Expected DataFrame, got {type(result)}"
            raise TypeError(msg)
        return result

    async def fetch_batch_sensor_data(
        self, requests: list[tuple[str, TimeWindow, list[str]]]
    ) -> dict[str, pd.DataFrame]:
        """Получить sensor data для multiple equipment (batch).

        Args:
            requests: List of (equipment_id, time_window, sensors) tuples

        Returns:
            results: Dictionary {equipment_id: DataFrame}

        Examples:
            >>> requests = [
            ...     ("exc_001", time_window1, ["pressure"]),
            ...     ("exc_002", time_window2, ["pressure", "temp"])
            ... ]
            >>> results = await connector.fetch_batch_sensor_data(requests)
            >>> results["exc_001"].shape
        """
        tasks = [
            self.fetch_sensor_data(eq_id, tw, sensors) for eq_id, tw, sensors in requests
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build result dict
        result_dict: dict[str, pd.DataFrame] = {}
        for (eq_id, _, _), result in zip(requests, results, strict=False):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch data for {eq_id}: {result}")
                result_dict[eq_id] = pd.DataFrame()  # Empty DataFrame
            else:
                result_dict[eq_id] = result

        return result_dict

    async def get_equipment_metadata(self, equipment_id: str) -> Any:
        """Получить equipment metadata.

        Args:
            equipment_id: Equipment identifier

        Returns:
            metadata: Equipment metadata (parsed to EquipmentMetadata when schema ready)

        Raises:
            ValueError: Если equipment не найден

        Examples:
            >>> metadata = await connector.get_equipment_metadata("exc_001")
            >>> print(metadata["equipment_type"])  # "excavator"
        """

        async def _fetch() -> Any:
            query = """
                SELECT
                    equipment_id,
                    equipment_type,
                    manufacturer,
                    model,
                    serial_number,
                    installation_date,
                    metadata_json
                FROM equipment_metadata
                WHERE equipment_id = $1
            """

            async with self.get_connection() as conn:
                row = await conn.fetchrow(query, equipment_id)

            if row is None:
                msg = f"Equipment not found: {equipment_id}"
                raise ValueError(msg)

            logger.info(f"Fetched metadata for {equipment_id}")
            return row

        result = await self._execute_with_retry(_fetch)
        if result is None:
            msg = f"Equipment not found: {equipment_id}"
            raise ValueError(msg)
        return result

    async def health_check(self) -> bool:
        """Проверить database connection health.

        Returns:
            healthy: True если подключение работает

        Examples:
            >>> is_healthy = await connector.health_check()
            >>> print(f"Database: {'UP' if is_healthy else 'DOWN'}")
        """
        if not self._connected or self.pool is None:
            return False

        try:
            async with self.get_connection() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.exception(f"Health check failed: {e}")
            return False

    async def get_equipment_count(self) -> int:
        """Получить количество equipment в database.

        Returns:
            count: Количество equipment records
        """

        async def _fetch() -> Any:
            query = "SELECT COUNT(*) FROM equipment_metadata"
            async with self.get_connection() as conn:
                return await conn.fetchval(query)

        result = await self._execute_with_retry(_fetch)
        if not isinstance(result, int):
            msg = f"Expected int, got {type(result)}"
            raise TypeError(msg)
        return result

    async def get_time_range(self, equipment_id: str) -> tuple[Any, Any]:
        """Получить time range доступных данных.

        Args:
            equipment_id: Equipment identifier

        Returns:
            time_range: (min_timestamp, max_timestamp)

        Examples:
            >>> start, end = await connector.get_time_range("exc_001")
            >>> print(f"Data available: {start} to {end}")
        """

        async def _fetch() -> tuple[Any, Any]:
            query = """
                SELECT
                    MIN(timestamp) as min_time,
                    MAX(timestamp) as max_time
                FROM sensor_data
                WHERE equipment_id = $1
            """

            async with self.get_connection() as conn:
                row = await conn.fetchrow(query, equipment_id)

            if row is None or row["min_time"] is None:
                msg = f"No data found for equipment: {equipment_id}"
                raise ValueError(msg)

            return (row["min_time"], row["max_time"])

        result = await self._execute_with_retry(_fetch)
        return result

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return (
            f"TimescaleConnector(status={status}, pool={self.pool_min_size}-{self.pool_max_size})"
        )
