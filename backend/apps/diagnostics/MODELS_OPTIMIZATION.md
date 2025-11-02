# Модели диагностики: оптимизации, индексы, денормализация и партиционирование

Документ описывает принятые решения по оптимизации моделей в приложении `diagnostics` и их влияние на производительность, сопровождаемые миграциями и сигналами.

## Цели
- Ускорить частые запросы (последние показания, списки систем/отчётов)
- Минимизировать N+1 через QuerySet-паттерны
- Подготовить хранилище к высокочастотным IoT-вставкам
- Обеспечить воспроизводимость схемы (миграции и чёткие имена индексов)

## Ключевые модели и изменения

### HydraulicSystem
- Денормализация: `components_count`, `last_reading_at`
- Индексы: `(status, system_type)`, `name`
- Валидация: `name != ''`, корректность `components_count` при `inactive`
- Менеджер: `with_prefetch()`, `active()`

### SystemComponent
- Уникальность имени в рамках системы: `Unique(system, name)`
- Индекс: `(system, name)`
- GIN для `specification` (через миграцию `RunSQL` c `jsonb_path_ops` при необходимости сложных запросов)

### SensorData
- Составные индексы: `(system, -timestamp)`, `(component, -timestamp)`, `(system, component, timestamp)`
- Сгенерированное поле: `day_bucket` (GeneratedField + TruncDay) для агрегатов
- Валидации: `unit != ''`, хотя бы одно из `value`/`value_decimal`
- QuerySet: `recent_for_system`, `for_component_range`, `with_system_component`
- Подготовка к партиционированию (TimescaleDB)

### DiagnosticReport
- Индексы: `(system, -created_at)`, `(severity, status)`
- Валидация: `ai_confidence` в диапазоне `[0,100]`
- QuerySet: `recent_for_system`, `open_critical`

## Сигналы (денормализация)
- При `SystemComponent` create/delete — обновляем `HydraulicSystem.components_count`
- При `SensorData` create — обновляем `HydraulicSystem.last_reading_at = GREATEST(old, new.timestamp)`

## Миграции
Рекомендуемая структура:

- `0002_optimize_models.py`
  - Индексы и constraints с фиксированными именами
  - GeneratedField `day_bucket`
  - (опционально) `RunSQL` для GIN индекса по JSON:
    ```sql
    CREATE INDEX IF NOT EXISTS component_spec_gin ON system_component USING GIN (specification jsonb_path_ops);
    ```

- `0003_enable_timescaledb.py`
  - Включение расширения и создание hypertable:
    ```sql
    CREATE EXTENSION IF NOT EXISTS timescaledb;
    SELECT create_hypertable('sensor_data', by_range('timestamp'), if_not_exists => TRUE);
    ```

- `0004_partition_management.py`
  - Доп. индексы/настройки под партиции при необходимости

Все имена индексов и ограничений заданы явно для воспроизводимости.

## Воспроизводимость
- При удалении БД: `python manage.py migrate` восстанавливает схему полностью
- При удалении БД и `migrations/`: `makemigrations` создаст декларативную схему; затем вручную активировать TimescaleDB командами из `0003_enable_timescaledb.py`

## Планы на Коммит 2
- Реализация TimescaleDB миграции и Celery задач для управления партициями:
  - `ensure_partitions_for_range(start, stop, '1 day')`
  - `cleanup_old_partitions('90 days')`
- Докеризация Postgres с TimescaleDB для dev/prod

## Замечания по производительности
- Для списков использовать `only()/defer()` и `select_related()/prefetch_related()`
- Контролировать batch-вставки данных датчиков (COPY/psycopg copy)
- Следить за autovacuum и индексами на партициях
