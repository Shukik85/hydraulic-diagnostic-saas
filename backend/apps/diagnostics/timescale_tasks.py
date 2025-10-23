"""
TimescaleDB задачи для манаджмента гипертаблиц и политик.

Оптимизировано под политики:
- Chunk interval: 7 дней
- Compression age: 30 дней
- Retention period: 365 дней
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from celery import shared_task
from django.db import connection
from django.utils import timezone

logger = logging.getLogger(__name__)

# Константы для соответствия с миграциями
DEFAULT_CHUNK_INTERVAL_DAYS: int = 7
DEFAULT_COMPRESSION_AGE_DAYS: int = 30
DEFAULT_RETENTION_PERIOD_DAYS: int = 365


@shared_task(bind=True, max_retries=3)
def ensure_partitions_for_range(
    self,  # type: ignore[no-untyped-def]
    table_name: str = "diagnostics_sensordata",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    chunk_interval: str = "7 days",
) -> Dict[str, Any]:
    """
    Обеспечивает создание chunk'ов TimescaleDB для указанного временного диапазона.
    """
    try:
        start_dt: datetime = (
            timezone.now()
            if start_time is None
            else datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        )
        end_dt: datetime = (
            start_dt + timedelta(days=30)
            if end_time is None
            else datetime.fromisoformat(end_time.replace("Z", "+00:00"))
        )

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = %s",
                [table_name],
            )
            if not cursor.fetchone():
                return {
                    "status": "failed",
                    "error": f"Table {table_name} is not a hypertable",
                    "task_id": getattr(self.request, "id", None),
                }

            cursor.execute(
                """
                SELECT chunk_name, range_start, range_end
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s
                AND range_start <= %s AND range_end >= %s
                ORDER BY range_start
                """,
                [table_name, end_dt, start_dt],
            )
            existing_chunks: List[tuple] = list(cursor.fetchall())

            cursor.execute(
                """
                SELECT d.interval_length
                FROM timescaledb_information.dimensions d
                JOIN timescaledb_information.hypertables h ON h.hypertable_name = d.hypertable_name
                WHERE h.hypertable_name = %s AND d.dimension_type = 'Time'
                """,
                [table_name],
            )
            interval_row = cursor.fetchone()

        result: Dict[str, Any] = {
            "status": "success",
            "table": table_name,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "existing_chunks": len(existing_chunks),
            "current_chunk_interval": (
                str(interval_row[0]) if interval_row else "unknown"
            ),
            "expected_chunk_interval": f"{DEFAULT_CHUNK_INTERVAL_DAYS} days",
            "task_id": getattr(self.request, "id", None),
        }
        return result

    except Exception as exc:  # noqa: BLE001
        logger.error("Partition creation failed: %s", exc)
        # В Celery .retry возвращает None; добавляем явный возврат для mypy
        try:
            self.retry(countdown=60, exc=exc)  # type: ignore[call-arg]
        finally:
            return {"status": "retry", "error": str(exc)}


@shared_task(bind=True, max_retries=2)
def cleanup_old_partitions(
    self,  # type: ignore[no-untyped-def]
    table_name: str = "diagnostics_sensordata",
    retention_period_days: int = DEFAULT_RETENTION_PERIOD_DAYS,
) -> Dict[str, Any]:
    """Удаляет старые чанки (старше retention_period_days)."""
    try:
        cutoff_time: datetime = timezone.now() - timedelta(days=retention_period_days)
        dropped_chunks: List[Dict[str, Any]] = []

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT format('%I.%I', chunk_schema, chunk_name) as full_chunk_name,
                       chunk_name, range_start, range_end,
                       pg_size_pretty(pg_total_relation_size(format('%I.%I', chunk_schema, chunk_name))) as size
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s AND range_end < %s
                ORDER BY range_start
                """,
                [table_name, cutoff_time],
            )
            for full_name, chunk_name, start_time, end_time, size in cursor.fetchall():
                try:
                    cursor.execute(
                        "SELECT drop_chunk(%s, if_exists => true)", [full_name]
                    )
                    dropped_chunks.append(
                        {
                            "name": chunk_name,
                            "range_start": start_time.isoformat(),
                            "range_end": end_time.isoformat(),
                            "size": size,
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to drop chunk %s: %s", chunk_name, e)

        return {
            "status": "success",
            "table": table_name,
            "retention_period_days": retention_period_days,
            "cutoff_time": cutoff_time.isoformat(),
            "chunks_dropped": len(dropped_chunks),
            "dropped_chunks": dropped_chunks,
            "task_id": getattr(self.request, "id", None),
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("Cleanup failed: %s", exc)
        try:
            self.retry(countdown=300, exc=exc)  # type: ignore[call-arg]
        finally:
            return {"status": "retry", "error": str(exc)}


@shared_task(bind=True)
def compress_old_chunks(
    self,  # type: ignore[no-untyped-def]
    table_name: str = "diagnostics_sensordata",
    compression_age_days: int = DEFAULT_COMPRESSION_AGE_DAYS,
) -> Dict[str, Any]:
    """Сжимает чанки старше compression_age_days."""
    try:
        cutoff: datetime = timezone.now() - timedelta(days=compression_age_days)
        compressed_chunks: List[Dict[str, Any]] = []

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT format('%I.%I', chunk_schema, chunk_name) as full_chunk_name,
                       chunk_name, range_start, range_end
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s AND range_end < %s AND NOT is_compressed
                ORDER BY range_start
                """,
                [table_name, cutoff],
            )
            for full_name, chunk_name, start_time, end_time in cursor.fetchall():
                try:
                    cursor.execute("SELECT compress_chunk(%s)", [full_name])
                    compressed_chunks.append(
                        {
                            "name": chunk_name,
                            "range_start": start_time.isoformat(),
                            "range_end": end_time.isoformat(),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to compress chunk %s: %s", chunk_name, e)

        return {
            "status": "success",
            "table": table_name,
            "compression_age_days": compression_age_days,
            "cutoff_time": cutoff.isoformat(),
            "chunks_compressed": len(compressed_chunks),
            "compressed_chunks": compressed_chunks,
            "task_id": getattr(self.request, "id", None),
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("Compression failed: %s", exc)
        return {
            "status": "failed",
            "error": str(exc),
            "task_id": getattr(self.request, "id", None),
        }


@shared_task
def get_hypertable_stats(table_name: str = "diagnostics_sensordata") -> Dict[str, Any]:
    """Получает статистику по hypertable."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT hypertable_size(%s) as total_size,
                       pg_size_pretty(hypertable_size(%s)) as total_size_pretty,
                       (SELECT COUNT(*) FROM timescaledb_information.chunks WHERE hypertable_name = %s) as total_chunks,
                       (SELECT COUNT(*) FROM timescaledb_information.chunks WHERE hypertable_name = %s AND is_compressed = true) as compressed_chunks
                """,
                [table_name, table_name, table_name, table_name],
            )
            total_size, total_size_pretty, total_chunks, compressed_chunks = (
                cursor.fetchone()
            )

            cursor.execute(
                """
                SELECT date_trunc('day', range_start) as day,
                       COUNT(*) as chunks_count,
                       pg_size_pretty(SUM(pg_total_relation_size(format('%I.%I', chunk_schema, chunk_name)))) as day_size
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s AND range_start >= NOW() - INTERVAL '30 days'
                GROUP BY date_trunc('day', range_start)
                ORDER BY day DESC
                LIMIT 10
                """,
                [table_name],
            )
            daily_stats = cursor.fetchall()

        return {
            "status": "success",
            "table": table_name,
            "total_size": total_size,
            "total_size_pretty": total_size_pretty,
            "total_chunks": int(total_chunks or 0),
            "compressed_chunks": int(compressed_chunks or 0),
            "compression_ratio": (
                f"{(int(compressed_chunks or 0) / int(total_chunks or 1) * 100):.1f}%"
                if int(total_chunks or 0) > 0
                else "0%"
            ),
            "current_policies": {
                "chunk_interval_days": DEFAULT_CHUNK_INTERVAL_DAYS,
                "compression_age_days": DEFAULT_COMPRESSION_AGE_DAYS,
                "retention_period_days": DEFAULT_RETENTION_PERIOD_DAYS,
            },
            "daily_stats": [
                {"date": d.isoformat() if d else None, "chunks": c, "size": s}
                for d, c, s in daily_stats
            ],
            "collected_at": timezone.now().isoformat(),
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("Stats collection failed: %s", exc)
        return {"status": "failed", "error": str(exc), "table": table_name}


@shared_task
def timescale_health_check() -> Dict[str, Any]:
    """Проверяет состояние TimescaleDB расширения и фоновых задач."""
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
            )
            version_row = cursor.fetchone()
            if not version_row:
                return {
                    "status": "failed",
                    "error": "TimescaleDB extension not installed",
                    "checked_at": timezone.now().isoformat(),
                }

            cursor.execute("SELECT COUNT(*) FROM timescaledb_information.hypertables")
            hypertables_count = int(cursor.fetchone()[0])

            cursor.execute(
                """
                SELECT application_name, proc_name, scheduled, config
                FROM timescaledb_information.jobs
                WHERE proc_name IN ('policy_compression', 'policy_retention')
                ORDER BY application_name
                """
            )
            active_policies = [
                {
                    "application_name": r[0],
                    "proc_name": r[1],
                    "scheduled": r[2],
                    "config": r[3],
                }
                for r in cursor.fetchall()
            ]

        return {
            "status": "success",
            "timescale_version": version_row[0],
            "hypertables_count": hypertables_count,
            "active_policies_count": len(active_policies),
            "active_policies": active_policies,
            "expected_policies": {
                "chunk_interval_days": DEFAULT_CHUNK_INTERVAL_DAYS,
                "compression_age_days": DEFAULT_COMPRESSION_AGE_DAYS,
                "retention_period_days": DEFAULT_RETENTION_PERIOD_DAYS,
            },
            "checked_at": timezone.now().isoformat(),
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("Health check failed: %s", exc)
        return {
            "status": "failed",
            "error": str(exc),
            "checked_at": timezone.now().isoformat(),
        }
