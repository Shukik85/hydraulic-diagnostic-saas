# GNN Service Monitoring

## Endpoints

- `/metrics` — Prometheus metrics
    - `gnn_inference_latency_seconds{model_version="..."}` — врачная задержка по версии
    - `gnn_inference_requests_total{model_version, status}` — число запросов (успешных/ошибок)
    - `gnn_active_models` — число активных версий
- `/health/live` — Liveness (жизнеспособность)
- `/health/ready` — Readiness (готовность к трафику)

## Пример интеграции с MLOps

1. Выбор версии модели через Registry
2. Запись детализированных метрик инференса
3. Проксирование latency/confidence в diagnosis_service (федеративный экспорт)

## Drift Detection

Для интеграции drift detection используйте методы из `diagnosis_service.mlops.drift_detector`:

```python
from diagnosis_service.mlops import get_drift_detector

# После инференса
model_version = "v2.1.0"
detector = get_drift_detector('gnn', model_version)
detector.add_production_sample(anomaly_score, confidence)

# Периодический drift check
score = detector.detect_drift()
if score > 0.3:
    print("⚠️ Model drift detected!")
```

## Checklist
- [x] Экспортировать latency, requests, errors прометей метрики
- [x] @app.get('/metrics') endpoint (через make_asgi_app)
- [x] Интегрировать версионность через model_registry
- [x] Использовать health endpoints для readiness/liveness в Kubernetes
- [x] Подключить drift detection к postprocessing инференса
