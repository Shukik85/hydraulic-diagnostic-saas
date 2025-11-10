# Generated migration for TimescaleDB hypertable

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("diagnostics", "0002_initial"),
    ]

    operations = [
        # 1. Удаляем старый Primary Key
        migrations.RunSQL(
            sql="""
            ALTER TABLE diagnostics_sensordata
            DROP CONSTRAINT IF EXISTS diagnostics_sensordata_pkey CASCADE;
            """,
            reverse_sql="""
            ALTER TABLE diagnostics_sensordata
            ADD CONSTRAINT diagnostics_sensordata_pkey PRIMARY KEY (id);
            """,
        ),
        # 2. Создаём составной Primary Key (id, timestamp)
        migrations.RunSQL(
            sql="""
            ALTER TABLE diagnostics_sensordata
            ADD CONSTRAINT diagnostics_sensordata_pkey
            PRIMARY KEY (id, timestamp);
            """,
            reverse_sql="""
            ALTER TABLE diagnostics_sensordata
            DROP CONSTRAINT diagnostics_sensordata_pkey;
            """,
        ),
        # 3. Создаём hypertable
        migrations.RunSQL(
            sql="""
            SELECT create_hypertable(
                'diagnostics_sensordata',
                'timestamp',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE,
                migrate_data => TRUE
            );
            """,
            reverse_sql="",  # Откат hypertable сложен
        ),
        # 4. Добавляем индексы ТОЛЬКО на timestamp (sensor_id добавим позже)
        migrations.RunSQL(
            sql="""
            CREATE INDEX IF NOT EXISTS idx_sensordata_timestamp
            ON diagnostics_sensordata (timestamp DESC);
            """,
            reverse_sql="""
            DROP INDEX IF EXISTS idx_sensordata_timestamp;
            """,
        ),
    ]
