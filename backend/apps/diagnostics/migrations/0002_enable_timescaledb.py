# Generated manually for TimescaleDB integration

from django.db import migrations


class Migration(migrations.Migration):
    """
    Миграция для включения TimescaleDB и создания hypertable для sensor_data.
    
    ВАЖНО: Эта миграция должна выполняться на пустой таблице sensor_data
    или после экспорта/импорта существующих данных.
    """
    
    dependencies = [
        ('diagnostics', '0001_initial'),
    ]

    operations = [
        # 1. Включаем расширение TimescaleDB (требует суперпользователя)
        migrations.RunSQL(
            sql="CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;",
            reverse_sql="-- DROP EXTENSION timescaledb CASCADE; -- Осторожно!",
        ),
        
        # 2. Преобразуем sensor_data в hypertable
        # Партиционирование по timestamp с интервалом 7 дней
        migrations.RunSQL(
            sql="""
                SELECT create_hypertable(
                    'sensor_data', 
                    by_range('timestamp'),
                    chunk_time_interval => INTERVAL '7 days',
                    if_not_exists => TRUE
                );
            """,
            reverse_sql="""
                -- Внимание: отмена преобразования в hypertable невозможна без потери данных
                -- SELECT drop_hypertable('sensor_data', if_exists => TRUE);
            """,
        ),
        
        # 3. Настройка сжатия данных старше 30 дней
        migrations.RunSQL(
            sql="""
                ALTER TABLE sensor_data SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'system_id, component_id',
                    timescaledb.compress_orderby = 'timestamp DESC'
                );
            """,
            reverse_sql="""
                ALTER TABLE sensor_data SET (timescaledb.compress = FALSE);
            """,
        ),
        
        # 4. Включаем автоматическое сжатие для старых chunk'ов
        migrations.RunSQL(
            sql="""
                SELECT add_compression_policy('sensor_data', INTERVAL '30 days');
            """,
            reverse_sql="""
                SELECT remove_compression_policy('sensor_data');
            """,
        ),
        
        # 5. Настройка политики очистки данных старше 1 года
        migrations.RunSQL(
            sql="""
                SELECT add_retention_policy('sensor_data', INTERVAL '1 year');
            """,
            reverse_sql="""
                SELECT remove_retention_policy('sensor_data');
            """,
        ),
    ]