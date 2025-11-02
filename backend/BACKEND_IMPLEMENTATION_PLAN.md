# BACKEND IMPLEMENTATION PLAN — Anomaly MVP (14 days)

Цель: end-to-end аномалии (ingestion → inference → API → UI) с целями p90 latency < 100ms и FPR < 10% на прод-валидации. Модели: RandomForest, XGBoost, HELM, Adaptive Thresholding.

## 1) Data Ingestion (TimescaleDB)
- Миграции: sensor_readings (system_id, equipment_id, sensor_id, ts timestamptz, v double, unit, meta JSONB)
- Hypertable + retention (5 лет), compression policy (LZ)
- Ingestion API: POST /api/v1/ingest (batch JSON/CSV), валидация, quarantine таблица
- Метрики: ingestion_rate, invalid_rows, protocol_health

## 2) ML Inference Service (FastAPI)
- Модели загружаются при старте (joblib/.json гиперпараметры)
- POST /api/v1/detect: вход 17 сенсоров; ответ {is_anomaly, score, affected_component, confidence, model_version}
- Ensemble: взвешенное голосование, feature flags на модели
- Профилирование и метрики: p90/p95 latency, error rate; Prometheus

## 3) Django Integration (DRF + Celery)
- App anomaly_detection: AnomalyResult(model, score, comp, severity, ts, system_id, equipment_id)
- Endpoints: 
  - GET /systems/{id}/anomalies?range&severity&page
  - GET /systems/{id}/anomaly-trend?range
  - POST /systems/{id}/analyze (sync/async)
- Celery задачи для пакетного анализа; retry с backoff, идемпотентность
- Кэш Redis для health/trend (TTL 30—60s)

## 4) Observability & Security
- Логи структурированы; correlation-id; Prometheus exporter
- Input validation строго по схемам; rate limit на POST /analyze
- Secrets в Vault/.env; параметризованные SQL

## 5) OpenAPI & Testing
- OpenAPI v1.0 для всех DRF/ingest/detect endpoints
- Unit: models/serializers/views; Интеграционные: DRF + FastAPI; perf: detect p90<100ms

## 6) Deployment
- docker-compose: django, fastapi, redis, timescale, prometheus
- Health-check endpoints; graceful shutdown; ресурсы контейнеров

## Чек‑лист готовности
- [ ] Hypertable/ingestion созданы, валидатор и quarantine работают
- [ ] FastAPI detect отдаёт ответ <100ms p90 на тестовом профиле
- [ ] DRF endpoints возвращают аномалии и тренды, кэш активен
- [ ] Метрики и логи видны в Prometheus/Grafana
- [ ] OpenAPI задокументирован, тесты зелёные