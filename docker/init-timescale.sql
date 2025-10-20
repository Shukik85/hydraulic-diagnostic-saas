-- Инициализация TimescaleDB для development окружения
-- Этот файл автоматически выполняется при создании контейнера с PostgreSQL

-- Создаем расширение TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Создаем дополнительные расширения для аналитики
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- Настройка базовых параметров для TimescaleDB
-- Увеличиваем work_mem для лучшей производительности аналитических запросов
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET max_connections = '200';

-- Настройки специфичные для TimescaleDB
ALTER SYSTEM SET timescaledb.max_background_workers = '8';

-- Применяем настройки (потребуется перезапуск PostgreSQL)
SELECT pg_reload_conf();

-- Создаем пользователя приложения (если нужно)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'app_user') THEN
        CREATE USER app_user WITH PASSWORD 'app_password';
        GRANT ALL PRIVILEGES ON DATABASE hydraulic_diagnostic TO app_user;
    END IF;
END
$$;

-- Сообщение о успешной инициализации
\echo 'TimescaleDB successfully initialized for hydraulic-diagnostic-saas!'
\echo 'Ready for hypertable creation through Django migrations.'