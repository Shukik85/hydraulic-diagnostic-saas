# Generated manually for TimescaleDB hypertable conversion
# Date: 2025-11-07 22:51 MSK

from django.db import migrations


class Migration(migrations.Migration):
    """
    Convert diagnostics_sensordata table to TimescaleDB hypertable.
    
    This migration:
    1. Creates hypertable partitioned by timestamp (7-day chunks)
    2. Adds compression policy (compress data older than 7 days)
    3. Adds retention policy (drop data older than 5 years)
    4. Optimizes for high-frequency sensor data ingestion
    
    Requirements:
    - TimescaleDB extension must be enabled in PostgreSQL
    - Table must already exist (created by 0001_initial.py)
    """

    dependencies = [
        ('diagnostics', '0002_initial'),
    ]

    operations = [
        # Step 1: Create hypertable from existing table
        # Partitioned by 'timestamp' column with 7-day chunk intervals
        migrations.RunSQL(
            sql="""
                SELECT create_hypertable(
                    'diagnostics_sensordata',
                    'timestamp',
                    chunk_time_interval => INTERVAL '7 days',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            """,
            reverse_sql="""
                -- Reverse: Cannot convert hypertable back to regular table safely
                -- Manual intervention required if rollback needed
                SELECT 1;
            """,
        ),
        
        # Step 2: Enable compression on the hypertable
        # Segment by sensor_type and system_id for optimal compression
        migrations.RunSQL(
            sql="""
                ALTER TABLE diagnostics_sensordata SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'sensor_type,system_id',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """,
            reverse_sql="""
                ALTER TABLE diagnostics_sensordata SET (
                    timescaledb.compress = false
                );
            """,
        ),
        
        # Step 3: Add compression policy
        # Compress chunks older than 7 days automatically
        migrations.RunSQL(
            sql="""
                SELECT add_compression_policy(
                    'diagnostics_sensordata',
                    INTERVAL '7 days',
                    if_not_exists => TRUE
                );
            """,
            reverse_sql="""
                SELECT remove_compression_policy(
                    'diagnostics_sensordata',
                    if_exists => TRUE
                );
            """,
        ),
        
        # Step 4: Add retention policy
        # Drop chunks older than 5 years (raw data retention)
        migrations.RunSQL(
            sql="""
                SELECT add_retention_policy(
                    'diagnostics_sensordata',
                    INTERVAL '5 years',
                    if_not_exists => TRUE
                );
            """,
            reverse_sql="""
                SELECT remove_retention_policy(
                    'diagnostics_sensordata',
                    if_exists => TRUE
                );
            """,
        ),
        
        # Step 5: Create continuous aggregate for hourly data (optional but recommended)
        migrations.RunSQL(
            sql="""
                CREATE MATERIALIZED VIEW IF NOT EXISTS diagnostics_sensordata_hourly
                WITH (timescaledb.continuous) AS
                SELECT
                    time_bucket('1 hour', timestamp) AS bucket,
                    system_id,
                    sensor_type,
                    COUNT(*) AS readings_count,
                    AVG(value) AS avg_value,
                    MIN(value) AS min_value,
                    MAX(value) AS max_value,
                    STDDEV(value) AS stddev_value
                FROM diagnostics_sensordata
                WHERE timestamp > NOW() - INTERVAL '90 days'
                GROUP BY bucket, system_id, sensor_type
                WITH NO DATA;
            """,
            reverse_sql="""
                DROP MATERIALIZED VIEW IF EXISTS diagnostics_sensordata_hourly;
            """,
        ),
        
        # Step 6: Add refresh policy for continuous aggregate
        migrations.RunSQL(
            sql="""
                SELECT add_continuous_aggregate_policy(
                    'diagnostics_sensordata_hourly',
                    start_offset => INTERVAL '3 hours',
                    end_offset => INTERVAL '1 hour',
                    schedule_interval => INTERVAL '1 hour',
                    if_not_exists => TRUE
                );
            """,
            reverse_sql="""
                SELECT remove_continuous_aggregate_policy(
                    'diagnostics_sensordata_hourly',
                    if_exists => TRUE
                );
            """,
        ),
    ]
