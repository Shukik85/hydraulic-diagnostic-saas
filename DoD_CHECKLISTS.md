# DoD Checklists per Roadmap Stage (Enterprise)

Ниже критерии приемки (Definition of Done) для каждого этапа обновлённого ROADMAP_INCREMENTAL.md.

## Этап 0. Frontend Foundation
- UI доступен локально, responsive (mobile-friendly), a11y базово
- Локализация RU/EN переключается в реальном времени
- Линтеры и pre-commit успешно проходят

## Этап 1. Data Ingestion & TimescaleDB
- Hypertables созданы; retention и compression включены (5 лет)
- Ingestion API принимает данные 10–60 сек, валидация и quarantine работают
- Метрики ingestion (rate, errors) экспортируются в Prometheus
- Контракты OpenAPI задокументированы

## Этап 2. ML Inference (4 модели)
- HELM/XGBoost/RandomForest/Adaptive интегрированы в pipeline
- p95 inference latency < 100ms (на тестовом профиле)
- Логи/метрики качества: accuracy, F1, FPR < 1%
- Фичефлаги на включение/отключение моделей

## Этап 3. Real-time Dashboard
- WebSocket обновляет gauges/charts/alerts timeline без фликов
- Heatmaps генерируются на агрегированных данных
- Equipment status синхронизирован с алертами
- e2e сценарий оператора проходит

## Этап 4. Predictive & RUL
- RUL генерируется и отображается в UI c доверительным интервалом
- Anomaly patterns и Cost calculator рассчитываются корректно
- Тесты на корректность lead time (2–6 часов)

## Этап 5. Reporting
- 40+ отчётов формируются по расписанию, PDF корректный
- Хранение в объектном сторе; доступ по pre-signed URL
- ROI/экономия считаются и отображаются

## Этап 6. Integrations
- Поддержка Slack/Telegram/Email/SMS/Jira/ServiceNow + Webhook API
- Health-check интеграций, ретраи с backoff
- Формы конфигураций в UI, валидация ключей

## Этап 7. Security & Compliance
- TLS 1.3, AES‑256, OAuth2/SAML2, MFA включены
- RBAC (admin/engineer/operator/viewer) применяется на API и UI
- Audit logs покрывают критичные действия, SOC2 отчёт доступен
- Backup & DR: RTO 1 час подтверждён

## Этап 8. Advanced ML
- Версионирование моделей, A/B‑тесты, онлайновое дообучение
- Explainability (feature importance, SHAP) доступно
- Тесты деградаций моделей и откатов

## Этап 9. Mobile
- Native iOS/Android: push уведомления, оффлайн режим
- Критичные сценарии доступны в мобильном UI

## Этап 10. Scaling & Performance
- Multi-tenant изоляция данных подтверждена
- Horizontal scaling включен, SLO: 99.9% uptime
- GraphQL API/Power BI коннектор доступны

---

Общие правила: API версии фиксированы; все изменения через feature flags; тесты (unit/integration/e2e) и наблюдаемость обязательны.