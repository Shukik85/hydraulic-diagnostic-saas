# Generated manually for TimescaleDB integration
# CI-compatible: gracefully handles missing TimescaleDB extension

from django.db import connection, migrations


def check_timescaledb_available():
    """
    Проверяет доступность TimescaleDB расширения.
    Возвращает True если расширение доступно и можно создавать hypertables.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb');"
            )
            return cursor.fetchone()[0]
    except Exception:
        return False


def check_table_is_hypertable(table_name: str = "sensor_data") -> bool:
    """
    Проверяет, является ли таблица hypertable.
    """
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS(
                    SELECT 1 FROM timescaledb_information.hypertables
                    WHERE hypertable_name = %s
                );
                """,
                [table_name],
            )
            return cursor.fetchone()[0]
    except Exception:
        return False


class Migration(migrations.Migration):
    """
    Миграция для включения TimescaleDB и создания hypertable для sensor_data.

    CI-совместимая: грациозно обрабатывает отсутствие расширения.
    В dev/prod окружении выполняет полную настройку TimescaleDB.
    """

    dependencies = [
        ("diagnostics", "0001_initial"),
    ]

    operations = [
        # 1. Включаем расширение TimescaleDB (требует суперпользователя)
        migrations.RunSQL(
            sql="CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;",
            reverse_sql="-- DROP EXTENSION timescaledb CASCADE; -- Осторожно!",
            state_operations=[],  # Не изменяет модели
        ),
        # 2. Преобразуем sensor_data в hypertable (если TimescaleDB доступен)
        migrations.RunSQL(
            sql="""
                DO $$
                BEGIN
                    -- Проверяем наличие TimescaleDB расширения
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                        -- Проверяем, не является ли таблица уже hypertable
                        IF NOT EXISTS(
                            SELECT 1 FROM timescaledb_information.hypertables
                            WHERE hypertable_name = 'sensor_data'
                        ) THEN
                            -- Создаем hypertable с 7-дневным партиционированием
                            PERFORM create_hypertable(
                                'sensor_data',
                                by_range('timestamp'),
                                chunk_time_interval => INTERVAL '7 days',
                                if_not_exists => TRUE
                            );
                            RAISE NOTICE 'TimescaleDB: sensor_data hypertable created successfully';
                        ELSE
                            RAISE NOTICE 'TimescaleDB: sensor_data is already a hypertable';
                        END IF;
                    ELSE
                        RAISE NOTICE 'TimescaleDB: Extension not available, skipping hypertable creation';
                    END IF;
                END $$;
            """,
            reverse_sql="""
                DO $$
                BEGIN
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
                        IF EXISTS(
                            SELECT 1 FROM timescaledb_information.hypertables
                            WHERE hypertable_name = 'sensor_data'
                        ) THEN
                            RAISE NOTICE 'TimescaleDB: Cannot automatically revert hypertable. Manual intervention required.';
                            -- Не можем автоматически откатить hypertable
                        END IF;
                    END IF;
                END $$;
            """,
            state_operations=[],
        ),
        # 3. Настройка сжатия данных (если hypertable создан)
        migrations.RunSQL(
            sql="""
                DO $$
                BEGIN
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') AND
                       EXISTS(SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_data') THEN
                        -- Настраиваем сжатие
                        ALTER TABLE sensor_data SET (
                            timescaledb.compress,
                            timescaledb.compress_segmentby = 'system_id, component_id',
                            timescaledb.compress_orderby = 'timestamp DESC'
                        );
                        RAISE NOTICE 'TimescaleDB: Compression settings configured';
                    ELSE
                        RAISE NOTICE 'TimescaleDB: Skipping compression settings (extension or hypertable not available)';
                    END IF;
                END $$;
            """,
            reverse_sql="""
                DO $$
                BEGIN
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') AND
                       EXISTS(SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_data') THEN
                        ALTER TABLE sensor_data SET (timescaledb.compress = FALSE);
                        RAISE NOTICE 'TimescaleDB: Compression disabled';
                    END IF;
                END $$;
            """,
            state_operations=[],
        ),
        # 4. Включаем автоматическое сжатие для старых chunk'ов
        migrations.RunSQL(
            sql="""
                DO $$
                BEGIN
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') AND
                       EXISTS(SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_data') THEN
                        -- Проверяем, нет ли уже политики сжатия
                        IF NOT EXISTS(
                            SELECT 1 FROM timescaledb_information.jobs
                            WHERE hypertable_name = 'sensor_data' AND proc_name = 'policy_compression'
                        ) THEN
                            PERFORM add_compression_policy('sensor_data', INTERVAL '30 days');
                            RAISE NOTICE 'TimescaleDB: Compression policy added (30 days)';
                        ELSE
                            RAISE NOTICE 'TimescaleDB: Compression policy already exists';
                        END IF;
                    ELSE
                        RAISE NOTICE 'TimescaleDB: Skipping compression policy (extension or hypertable not available)';
                    END IF;
                END $$;
            """,
            reverse_sql="""
                DO $$
                BEGIN
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') AND
                       EXISTS(SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_data') THEN
                        PERFORM remove_compression_policy('sensor_data', if_exists => TRUE);
                        RAISE NOTICE 'TimescaleDB: Compression policy removed';
                    END IF;
                END $$;
            """,
            state_operations=[],
        ),
        # 5. Настройка политики очистки данных
        migrations.RunSQL(
            sql="""
                DO $$
                BEGIN
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') AND
                       EXISTS(SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_data') THEN
                        -- Проверяем, нет ли уже политики очистки
                        IF NOT EXISTS(
                            SELECT 1 FROM timescaledb_information.jobs
                            WHERE hypertable_name = 'sensor_data' AND proc_name = 'policy_retention'
                        ) THEN
                            PERFORM add_retention_policy('sensor_data', INTERVAL '1 year');
                            RAISE NOTICE 'TimescaleDB: Retention policy added (1 year)';
                        ELSE
                            RAISE NOTICE 'TimescaleDB: Retention policy already exists';
                        END IF;
                    ELSE
                        RAISE NOTICE 'TimescaleDB: Skipping retention policy (extension or hypertable not available)';
                    END IF;
                END $$;
            """,
            reverse_sql="""
                DO $$
                BEGIN
                    IF EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') AND
                       EXISTS(SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'sensor_data') THEN
                        PERFORM remove_retention_policy('sensor_data', if_exists => TRUE);
                        RAISE NOTICE 'TimescaleDB: Retention policy removed';
                    END IF;
                END $$;
            """,
            state_operations=[],
        ),
    ]
