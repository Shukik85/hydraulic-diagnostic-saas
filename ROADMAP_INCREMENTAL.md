# Roadmap: Инкрементальная реализация Hydraulic Diagnostic SaaS (Working Software After Every Step)

Цель: построить план, где после каждого шага проект остаётся работоспособным, а следующий шаг добавляет функциональность. Приоритеты: безопасность, стабильность, наблюдаемость. Все изменения — малыми атомарными диффами, строго по .pre-commit-config.yaml.

## Этап 0. Базовая среда и наблюдаемость (день 0)
- Стабилизировать фронтенд: Nuxt 4 + Tailwind v4 (готово), единый premium-tokens.css.
- Настроить postcss ('@tailwindcss/postcss', autoprefixer) и dev middleware auth stub (в dev пропускает auth).
- Включить Sentry/Logs dev-only (моки): сбор ошибок на фронте, консольная телеметрия.
- Smoke-тесты backend (pytest -k smoke). Docker Compose base с Redis, Postgres/Timescale, MinIO (пустые контейнеры).
Результат: UI доступен локально, базовые контейнеры поднимаются, логи пишутся.

## Этап 1. Аутентификация как фундамент (неделя 1)
Зависимости: Этап 0
- Backend: JWT (access/refresh), роли (ADMIN/ENGINEER/VIEWER), сессии, rate-limit на /auth.
- Frontend: форма входа подключена к real /auth/login, хранение токенов (httpOnly cookie/secure), logout.
- Middleware: защита приватных маршрутов, dev-байпас сохраняем в dev.
- Тесты: unit (users), e2e вход/выход, негативные сценарии (lockout/CAPTCHA-заглушка).
Результат: проект требует входа и работает полностью после логина.

## Этап 2. Каркас данных и метрики дашборда (неделя 2)
Зависимости: Этап 1
- Backend: модели equipment/systems, базовые метрики (active_systems, total_diagnostics) с fake источником.
- API: /dashboard/metrics (кэш Redis 30с), /systems/list.
- Frontend: дашборд подключён к API, KPI показывают реальные числа, заглушки заменены.
- Наблюдаемость: метрики API latency, error rate, логи запросов.
Результат: авторизованный пользователь видит живые KPI с бэкенда.

## Этап 3. Минимальный Diagnostics Engine (MVP) (неделя 3)
Зависимости: Этап 2
- Backend: DiagnosticSession (создание/статусы), Celery task run_diagnostic (mock вычисления), прогресс в Redis.
- API: POST /diagnostics/run, GET /diagnostics/:id, CANCEL.
- Frontend: страницы Diagnostics подключены: запуск, прогресс, завершение; Recent Results из API.
- WebSocket: канал только для прогресса (Channels), graceful reconnect.
Результат: можно запустить диагностику и увидеть её завершение.

## Этап 4. Сенсорные данные: ingest и таблица (неделя 4)
Зависимости: Этап 3
- Backend: TimescaleDB hypertable sensor_readings, ingestion API (CSV upload) с валидацией и quarantine.
- API: POST /sensors/upload, GET /sensors/readings?range/equipment, пагинация.
- Frontend: Sensors страница читает реальные данные, фильтры и пагинация работают.
- Тесты: performance на чтение 50k строк, валидация CSV.
Результат: можно загружать и просматривать сенсорные данные.

## Этап 5. Графики и агрегации (неделя 5)
Зависимости: Этап 4
- Backend: continuous aggregates (hourly/daily), downsampling, кэширование ответов графиков.
- API: /charts/pressure, /charts/temperature, /charts/flow (time-range).
- Frontend: графики VeCharts подключены, zoom/pan, пустые состояния.
Результат: дашборд и Sensors показывают живые графики из TimescaleDB.

## Этап 6. Reports: генерация PDF (неделя 6)
Зависимости: Этапы 3–5
- Backend: ReportTemplate/GeneratedReport, генерация PDF (WeasyPrint/ReportLab), хранение в MinIO, подпись хешем.
- API: POST /reports/generate (по сессии диагностики), GET /reports/:id/download (pre-signed URL).
- Frontend: Reports страница — генерация и скачивание, статусы готовности.
Результат: можно генерировать и скачивать брендированные отчёты.

## Этап 7. Alerts и пороги (неделя 7)
Зависимости: Этапы 4–5
- Backend: AlertThreshold, движок триггеров (hysteresis), уведомления (email/Slack моки), подавление флаппинга.
- API: CRUD thresholds, список алертов, acknowledge.
- Frontend: управление порогами, бейджи статусов, acknowledge в UI.
Результат: система умеет сигнализировать о выходе показателей за пороги.

## Этап 8. RAG Assistant (минимально полезный) (неделя 8)
Зависимости: Этап 1 (auth)
- Backend: загрузка docs, построение FAISS, поиск top‑k, локальная LLM через Ollama, логирование токенов.
- API: /rag/ask, /rag/docs (ingest/list), базовые ACL на документы.
- Frontend: Chat подключён к /rag/ask, отображение источников, экспорт чата в Markdown.
Результат: ассистент отвечает на основе загруженных документов.

## Этап 9. Интеграции (SCADA/ERP) — адаптерный слой (неделя 9)
Зависимости: Этапы 4–7
- Backend: адаптеры Modbus/OPC‑UA (конфигурации + health-check), webhook-и для ERP/CMMS, маппинг полей.
- Безопасность: хранение кредов в Vault (или шифр. поле), rate-limit и retry с backoff.
- Frontend: формы интеграций, статусы, тест подключения.
Результат: базовое подключение внешних систем, план дальнейшего расширения.

## Этап 10. Production hardening (неделя 10)
Зависимости: все предыдущие
- Security: MFA, audit trail, field-level encryption PII, CSP/Headers, зависимостной скан.
- Observability: Prometheus + OpenTelemetry, дашборды, алерты SLO.
- Perf: кэширование горячих эндпоинтов, индексы БД, профилирование; chaos-инструменты на стейдже.
- CI/CD: blue‑green/canary, миграции БД, seed, rollback сценарии.
Результат: готовность к стабильной эксплуатации и масштабированию.

---

## Правила инкремента
- После каждого этапа: проект запускается локально (docker compose up) и доступен end-to-end.
- API контракты фиксируются OpenAPI, изменения — через версионирование.
- Feature flags для опасных фич; dev заглушки отключаются в prod.
- Все новые фичи покрываются тестами и наблюдаемостью (метрики/логи/трейсы).

## Матрица зависимостей (упрощённо)
- Auth → Metrics → Diagnostics → Sensors → Charts → Reports → Alerts → RAG → Integrations → Prod Hardening
- Каждый шаг использует предыдущие API/данные, не ломая обратную совместимость.

## Checkpoints после каждого этапа
- Demo сценарий (скрипт действий) и чек-лист регрессии.
- Бюджет производительности: p95 latency целевых эндпоинтов.
- Security checklist: токены, роли, логи, секреты.
- UX checklist: пустые состояния, загрузка, ошибки, доступность (a11y).
