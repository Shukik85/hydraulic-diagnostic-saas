# RAG Service Monitoring

## Endpoints

- `/metrics` — Prometheus метрики
    - `rag_generation_latency_seconds{model_version}` — время генерации интерпретации
    - `rag_requests_total{model_version, status}` — число успешных/ошибочных запросов
    - `rag_active_models` — число активных версий RAG моделей
    - `rag_tokens_used` — токены на запрос (драйвер расходов)
- `/health`, `/ready` — liveness/readiness

## Пример интеграции с MLOps (diagnosis_service.mlops)

```python
from diagnosis_service.mlops import model_registry, get_drift_detector

# Выбор актуальной версии в инференсе
model_version = model_registry.get_champion('rag').version

# После генерации интерпретации:
confidence = 0.87
processing_time = 1.97  # (сек)
drift_detector = get_drift_detector('rag', model_version)
drift_detector.add_production_sample(confidence, confidence)  # Тут удобно передавать одно и то же

# Обновление в Prometheus
from prometheus_client import Histogram
rag_generation_latency = Histogram('rag_generation_latency_seconds', 'RAG generation', ['model_version'])
rag_generation_latency.labels(model_version=model_version).observe(processing_time)
```

## Drift Detection и Reporting

- Добавляй prediction/confidence всех успешных вызовов в drift detector (см. diagnosis_service.mlops.drift_detector)
- Периодически вызывай `drift_detector.detect_drift()`, логируй score: если >0.3 — алертация/уведомление
- Экспонируется как `model_drift_score{model_type="rag", model_version=...}`

## Checklist
- [x] Экспортировать latency/requests/tokens метрики Prometheus
- [x] Поддерживать health/ready endpoint
- [x] Интегрировать версионность через model_registry
- [x] Подключить drift detection (как в gnn_service)
- [x] Документировать все эндпойнты и обновления. 
