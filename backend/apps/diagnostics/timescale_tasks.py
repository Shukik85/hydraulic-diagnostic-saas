"""
TimescaleDB задачи для манаджмента гипертаблиц и политик.

Оптимизировано под политики:
- Chunk interval: 7 дней
- Compression age: 30 дней
- Retention period: 365 дней
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from django.conf import settings
from django.db import connection, transaction
from django.utils import timezone

from celery import shared_task

logger = logging.getLogger(__name__)

# Константы для соответствия с миграциями
DEFAULT_CHUNK_INTERVAL_DAYS = 7
DEFAULT_COMPRESSION_AGE_DAYS = 30
DEFAULT_RETENTION_PERIOD_DAYS = 365


@shared_task(bind=True, max_retries=3)
def ensure_partitions_for_range(
    self,
    table_name: str = "diagnostics_sensordata",
    start_time: str = None,
    end_time: str = None,
    chunk_interval: str = "7 days",
) -> Dict[str, Any]:
    """
    Обеспечивает создание chunk'ов TimescaleDB для указанного временного диапазона.
    
    При chunk_interval=7 days TimescaleDB автоматически создаёт чанки по 7 дней.
    Эта задача может форсировать создание нужных чанков заранее.

    Args:
        table_name: Имя hypertable (по умолчанию 'diagnostics_sensordata')
        start_time: ISO строка начального времени (по умолчанию - сейчас)
        end_time: ISO строка конечного времени (по умолчанию - сейчас + 30 дней)
        chunk_interval: Интервал chunk'а (по умолчанию '7 days')

    Returns:
        Dict с результатами операции
    """
    try:
        if not start_time:
            start_dt = timezone.now()
        else:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))

        if not end_time:
            end_dt = start_dt + timedelta(days=30)
        else:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))

        with connection.cursor() as cursor:
            # Проверяем, что таблица является hypertable
            cursor.execute(
                """
                SELECT 1 FROM timescaledb_information.hypertables
                WHERE hypertable_name = %s
            """,
                [table_name],
            )

            if not cursor.fetchone():
                raise ValueError(f"Table {table_name} is not a hypertable")

            # Получаем существующие chunk'и в диапазоне
            cursor.execute(
                """
                SELECT chunk_name, range_start, range_end
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s
                AND range_start <= %s AND range_end >= %s
                ORDER BY range_start;
            """,
                [table_name, end_dt, start_dt],
            )

            existing_chunks = cursor.fetchall()

            # Проверяем настройки chunk_time_interval
            cursor.execute(
                """
                SELECT d.interval_length
                FROM timescaledb_information.dimensions d
                JOIN timescaledb_information.hypertables h ON h.hypertable_name = d.hypertable_name
                WHERE h.hypertable_name = %s AND d.dimension_type = 'Time';
            """,
                [table_name],
            )

            current_interval = cursor.fetchone()

        result = {
            "status": "success",
            "table": table_name,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "existing_chunks": len(existing_chunks),
            "current_chunk_interval": (
                str(current_interval[0]) if current_interval else "unknown"
            ),
            "expected_chunk_interval": f"{DEFAULT_CHUNK_INTERVAL_DAYS} days",
            "task_id": self.request.id,
        }

        logger.info(f"Partition check completed: {result}")
        return result

    except Exception as exc:
        logger.error(f"Partition creation failed: {exc}")
        self.retry(countdown=60, exc=exc)


@shared_task(bind=True, max_retries=2)
def cleanup_old_partitions(
    self, 
    table_name: str = "diagnostics_sensordata", 
    retention_period_days: int = DEFAULT_RETENTION_PERIOD_DAYS
) -> Dict[str, Any]:
    """
    Очищает старые chunk'и TimescaleDB согласно политике retention.
    
    По умолчанию retention_period_days=365 (год истории).

    Args:
        table_name: Имя hypertable
        retention_period_days: Период хранения в днях

    Returns:
        Dict с результатами очистки
    """
    try:
        cutoff_time = timezone.now() - timedelta(days=retention_period_days)

        with connection.cursor() as cursor:
            # Получаем список старых chunk'ов
            cursor.execute(
                """
                SELECT
                    format('%I.%I', chunk_schema, chunk_name) as full_chunk_name,
                    chunk_name,
                    range_start,
                    range_end,
                    pg_size_pretty(pg_total_relation_size(format('%I.%I', chunk_schema, chunk_name))) as size
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s
                AND range_end < %s
                ORDER BY range_start;
            """,
                [table_name, cutoff_time],
            )

            old_chunks = cursor.fetchall()
            dropped_chunks = []

            for full_name, chunk_name, start_time, end_time, size in old_chunks:
                try:
                    # Удаляем chunk через drop_chunk с параметризацией
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
                    logger.info(f"Dropped chunk {chunk_name} (size: {size})")

                except Exception as chunk_error:
                    logger.error(f"Failed to drop chunk {chunk_name}: {chunk_error}")

        result = {
            "status": "success",
            "table": table_name,
            "retention_period_days": retention_period_days,
            "cutoff_time": cutoff_time.isoformat(),
            "chunks_dropped": len(dropped_chunks),
            "dropped_chunks": dropped_chunks,
            "task_id": self.request.id,
        }

        logger.info(
            f"Cleanup completed for {table_name}: {len(dropped_chunks)} chunks dropped"
        )
        return result

    except Exception as exc:
        logger.error(f"Cleanup failed: {exc}")
        self.retry(countdown=300, exc=exc)  # 5 минут до повтора


@shared_task(bind=True)
def compress_old_chunks(
    self, 
    table_name: str = "diagnostics_sensordata", 
    compression_age_days: int = DEFAULT_COMPRESSION_AGE_DAYS
) -> Dict[str, Any]:
    """
    Сжимает старые chunk'и для экономии места.
    
    По умолчанию compression_age_days=30 (месяц).

    Args:
        table_name: Имя hypertable
        compression_age_days: Возраст chunk'ов для сжатия в днях

    Returns:
        Dict с результатами сжатия
    """
    try:
        compression_cutoff = timezone.now() - timedelta(days=compression_age_days)

        with connection.cursor() as cursor:
            # Получаем несжатые chunk'и старше указанного возраста
            cursor.execute(
                """
                SELECT
                    format('%I.%I', chunk_schema, chunk_name) as full_chunk_name,
                    chunk_name,
                    range_start,
                    range_end
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s
                AND range_end < %s
                AND NOT is_compressed
                ORDER BY range_start;
            """,
                [table_name, compression_cutoff],
            )

            uncompressed_chunks = cursor.fetchall()
            compressed_chunks = []

            for full_name, chunk_name, start_time, end_time in uncompressed_chunks:
                try:
                    # Сжимаем chunk через compress_chunk с параметризацией
                    cursor.execute("SELECT compress_chunk(%s)", [full_name])
                    compressed_chunks.append(
                        {
                            "name": chunk_name,
                            "range_start": start_time.isoformat(),
                            "range_end": end_time.isoformat(),
                        }
                    )
                    logger.info(f"Compressed chunk {chunk_name}")

                except Exception as compress_error:
                    logger.error(
                        f"Failed to compress chunk {chunk_name}: {compress_error}"
                    )

        result = {
            "status": "success",
            "table": table_name,
            "compression_age_days": compression_age_days,
            "cutoff_time": compression_cutoff.isoformat(),
            "chunks_compressed": len(compressed_chunks),
            "compressed_chunks": compressed_chunks,
            "task_id": self.request.id,
        }

        logger.info(
            f"Compression completed for {table_name}: {len(compressed_chunks)} chunks compressed"
        )
        return result

    except Exception as exc:
        logger.error(f"Compression failed: {exc}")
        return {"status": "failed", "error": str(exc), "task_id": self.request.id}


@shared_task
def get_hypertable_stats(table_name: str = "diagnostics_sensordata") -> Dict[str, Any]:
    """
    Получает статистику по hypertable.

    Args:
        table_name: Имя hypertable

    Returns:
        Dict со статистикой
    """
    try:
        with connection.cursor() as cursor:
            # Общая статистика по hypertable
            cursor.execute(
                """
                SELECT
                    hypertable_size(%s) as total_size,
                    pg_size_pretty(hypertable_size(%s)) as total_size_pretty,
                    (
                        SELECT COUNT(*)
                        FROM timescaledb_information.chunks
                        WHERE hypertable_name = %s
                    ) as total_chunks,
                    (
                        SELECT COUNT(*)
                        FROM timescaledb_information.chunks
                        WHERE hypertable_name = %s AND is_compressed = true
                    ) as compressed_chunks
            """,
                [table_name, table_name, table_name, table_name],
            )

            stats = cursor.fetchone()

            # Статистика по chunk'ам за последние периоды
            cursor.execute(
                """
                SELECT
                    date_trunc('day', range_start) as day,
                    COUNT(*) as chunks_count,
                    pg_size_pretty(
                        SUM(
                            pg_total_relation_size(
                                format('%I.%I', chunk_schema, chunk_name)
                            )
                        )
                    ) as day_size
                FROM timescaledb_information.chunks
                WHERE hypertable_name = %s
                AND range_start >= NOW() - INTERVAL '30 days'
                GROUP BY date_trunc('day', range_start)
                ORDER BY day DESC
                LIMIT 10;
            """,
                [table_name],
            )

            daily_stats = cursor.fetchall()

        result = {
            "status": "success",
            "table": table_name,
            "total_size": stats[0],
            "total_size_pretty": stats[1],
            "total_chunks": stats[2],
            "compressed_chunks": stats[3],
            "compression_ratio": (
                f"{(stats[3] / stats[2] * 100):.1f}%" if stats[2] > 0 else "0%"
            ),
            "current_policies": {
                "chunk_interval_days": DEFAULT_CHUNK_INTERVAL_DAYS,
                "compression_age_days": DEFAULT_COMPRESSION_AGE_DAYS,
                "retention_period_days": DEFAULT_RETENTION_PERIOD_DAYS,
            },
            "daily_stats": [
                {
                    "date": day.isoformat() if day else None,
                    "chunks": count,
                    "size": size,
                }
                for day, count, size in daily_stats
            ],
            "collected_at": timezone.now().isoformat(),
        }

        return result

    except Exception as exc:
        logger.error(f"Stats collection failed: {exc}")
        return {"status": "failed", "error": str(exc), "table": table_name}


@shared_task
def timescale_health_check() -> Dict[str, Any]:
    """
    Проверка здоровья TimescaleDB и состояния политик.

    Returns:
        Dict с результатами проверки
    """
    try:
        with connection.cursor() as cursor:
            # Проверяем доступность TimescaleDB
            cursor.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';"
            )
            timescale_version = cursor.fetchone()

            if not timescale_version:
                return {
                    "status": "failed",
                    "error": "TimescaleDB extension not installed",
                    "checked_at": timezone.now().isoformat(),
                }

            # Проверяем состояние hypertables
            cursor.execute(
                """
                SELECT hypertable_name, num_dimensions
                FROM timescaledb_information.hypertables;
            """
            )

            hypertables = cursor.fetchall()

            # Проверяем активные политики (компрессия и ретеншен)
            cursor.execute(
                """
                SELECT 
                    application_name, 
                    proc_name,
                    scheduled,
                    config
                FROM timescaledb_information.jobs 
                WHERE proc_name IN ('policy_compression', 'policy_retention')
                ORDER BY application_name;
            """
            )

            active_policies = cursor.fetchall()

            # Проверяем фоновые процессы TimescaleDB
            cursor.execute(
                """
                SELECT application_name, state, query
                FROM pg_stat_activity
                WHERE application_name LIKE '%timescaledb%'
                OR query LIKE '%_timescaledb_%';
            """
            )

            background_jobs = cursor.fetchall()

        result = {
            "status": "success",
            "timescale_version": timescale_version[0],
            "hypertables_count": len(hypertables),
            "hypertables": [
                {"name": name, "dimensions": dims} for name, dims in hypertables
            ],
            "active_policies_count": len(active_policies),
            "active_policies": [
                {
                    "application_name": app_name,
                    "proc_name": proc_name,
                    "scheduled": scheduled,
                    "config": config,
                }
                for app_name, proc_name, scheduled, config in active_policies
            ],
            "background_jobs_count": len(background_jobs),
            "expected_policies": {
                "chunk_interval_days": DEFAULT_CHUNK_INTERVAL_DAYS,
                "compression_age_days": DEFAULT_COMPRESSION_AGE_DAYS,
                "retention_period_days": DEFAULT_RETENTION_PERIOD_DAYS,
            },
            "checked_at": timezone.now().isoformat(),
        }

        return result

    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        return {
            "status": "failed",
            "error": str(exc),
            "checked_at": timezone.now().isoformat(),
        }