# Generated migration for TimescaleDB optimization
from django.db import migrations


class Migration(migrations.Migration):
    """
    Миграция для оптимизации под TimescaleDB:
    - Создание hypertable для sensor_data
    - Установка политик compression и retention
    - Chunk interval = 7 days
    """

    dependencies = [
        ("diagnostics", "0001_initial"),
    ]

    operations = [
        # TimescaleDB hypertable setup
        migrations.RunSQL(
            sql=[
                # Создание hypertable с защитой от повторного выполнения
                "SELECT create_hypertable('diagnostics_sensordata', 'timestamp', if_not_exists => TRUE);",
                # Установка chunk interval = 7 дней
                "ALTER TABLE diagnostics_sensordata SET (timescaledb.chunk_time_interval = INTERVAL '7 days');",
            ],
            reverse_sql=[
                # В случае rollback - возвращаем к обычной таблице (необратимо без потери данных)
                "-- TimescaleDB hypertable cannot be easily reverted without data loss",
            ],
        ),
        # Политика сжатия (30 дней)
        migrations.RunSQL(
            sql=[
                "SELECT add_compression_policy('diagnostics_sensordata', INTERVAL '30 days', if_not_exists => TRUE);"
            ],
            reverse_sql=[
                "SELECT remove_compression_policy('diagnostics_sensordata', if_exists => TRUE);"
            ],
        ),
        # Политика retention (365 дней)
        migrations.RunSQL(
            sql=[
                "SELECT add_retention_policy('diagnostics_sensordata', INTERVAL '365 days', if_not_exists => TRUE);"
            ],
            reverse_sql=[
                "SELECT remove_retention_policy('diagnostics_sensordata', if_exists => TRUE);"
            ],
        ),
        # GIN индекс для JSONB specification в SystemComponent
        migrations.RunSQL(
            sql=[
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_comp_specification_gin ON diagnostics_systemcomponent USING gin (specification);"
            ],
            reverse_sql=["DROP INDEX IF EXISTS idx_comp_specification_gin;"],
        ),
        # Статистика для оптимизатора запросов TimescaleDB
        migrations.RunSQL(
            sql=[
                "ANALYZE diagnostics_sensordata;",
                "ANALYZE diagnostics_hydraulicsystem;",
                "ANALYZE diagnostics_systemcomponent;",
            ],
            reverse_sql=["-- No reverse for ANALYZE"],
        ),
    ]