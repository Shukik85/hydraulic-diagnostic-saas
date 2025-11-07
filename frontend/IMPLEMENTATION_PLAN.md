# IMPLEMENTATION PLAN — Nuxt Frontend (Enterprise Anomaly MVP)

Цель: за 14 дней доставить end-to-end поток аномалий (ingestion → inference → API → UI). Сохранить инкрементальность, единый layout, RU/EN локализацию.

## Страницы и маршруты
- /systems/index — список систем (готово)
- /systems/[id] — карточка системы (контент уточняется)
- /systems/[id]/sensors/index — список датчиков (пустые состояния)
- /systems/[id]/equipments/index — список оборудования (+ добавление)
- /diagnostics — мониторинг диагностик (подключение позже)
- /reports — отчёты (подключение позже)
- /settings — единая страница с вкладками (оставляем, i18n реактивный)

## Компоненты (первый набор)
- AnomalyChart — графики трендов аномалий (подключение к API этап 3)
- ComponentStatus — светофор по компонентам (GREEN/YELLOW/RED)
- AlertPanel — список алертов с severity и acknowledge
- UCreateEquipmentModal — модалка добавления оборудования (валидация, i18n)
- UCreateSensorModal — модалка добавления датчика (валидация, i18n)

## Состояния и UX
- Пустые состояния для sensors/equipments/alerts с CTA кнопками
- Skeleton loaders при загрузке списков/графиков
- Тосты для успеха/ошибок (унифицированные стили)

## i18n
- Ключи: sensors.*, equipments.*, anomalies.*, alerts.*
- Реактивные вкладки: через computed(() => ...) для t()
- Исключить HTML/Markdown из json переводов

## API интеграции (последовательно)
1) Моки (фронт): плейсхолдеры списков/графиков
2) Подключение к DRF: GET /systems/{id}/equipments, POST /equipments (создание)
3) Подключение к аномалиям: GET /systems/{id}/anomalies, GET /trend
4) Acknowledge: POST /alerts/{id}/ack

## Тестирование
- e2e сценарий: открыть систему → добавить оборудование → видеть пустой список аномалий и тренды → без ошибок и i18n предупреждений
- Unit: базовая валидация форм модалок

## Наблюдаемость
- Логи ошибок в консоль, включённые sourcemaps
- В prod включить Sentry (после backend готовности)

## Порядок работ (атомарно)
1) Каркас equipments/index.vue + модалка добавления (без API)
2) Аналогично sensors/index.vue + модалка добавления (без API)
3) Подключение GET оборудований/датчиков (после DRF)
4) Добавить AnomalyChart + заглушка
5) Подключить GET anomalies/trend
6) AlertPanel + acknowledge (после DRF)

## Ограничения
- Не создавать синтетические прод-данные, только UI пустые состояния
- Не дублировать layouts, соблюдать единый dashboard layout
- Строгий контроль i18n — без ошибок в консоли