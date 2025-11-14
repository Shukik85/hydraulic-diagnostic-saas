# MLOps Module

Модуль для управления ML моделями в production: versioning, A/B testing, drift detection.

## 📦 Структура

```
mlops/
├── __init__.py           # Экспорты
├── versioning.py        # Model Registry + Model Version
├── ab_testing.py        # A/B Testing Framework
├── drift_detector.py    # Drift Detection
└── README.md            # Этот файл
```

---

## 1️⃣ Model Versioning

### Использование

```python
from mlops import model_registry, ModelVersion

# Зарегистрировать новую версию GNN модели
new_gnn = ModelVersion(
    model_type='gnn',
    version='v2.2.0',
    model_path='/models/gnn_v2.2.0.pt',
    config={
        'architecture': 'GraphSAGE',
        'hidden_dim': 256,  # Increased
        'num_layers': 4
    }
)

model_registry.register_version(new_gnn)

# Получить текущую production модель
champion = model_registry.get_champion('gnn')
print(f"Current GNN: {champion.version}")

# Обновить метрики после inference
model_registry.update_metrics(
    model_type='gnn',
    version='v2.1.0',
    inference_time_ms=124.5,
    confidence=0.87,
    error=False
)
```

### API

#### `ModelVersion`
```python
@dataclass
class ModelVersion:
    model_type: str          # 'gnn' | 'rag'
    version: str             # 'v2.1.0'
    is_champion: bool        # Production model?
    traffic_percentage: int  # 0-100 (for A/B testing)
    model_path: str          # Path to model file
    config: Dict             # Model config
    
    # Performance metrics
    avg_inference_time_ms: float
    avg_confidence: float
    error_rate: float
```

#### `ModelRegistry`
- `register_version(version: ModelVersion)` - зарегистрировать новую версию
- `get_champion(model_type: str)` - получить champion
- `get_version_for_request(model_type, user_id)` - выбрать версию для A/B test
- `update_metrics(...)` - обновить метрики
- `promote_to_champion(model_type, version)` - промоутить в production

---

## 2️⃣ A/B Testing

### Пример: Тестирование новой GNN модели

```python
from mlops import ab_test_manager, ABTestConfig

# 1. Запустить A/B тест
test_config = ABTestConfig(
    name='gnn_v2.2_test',
    model_type='gnn',
    control_version='v2.1.0',      # Текущая production
    treatment_version='v2.2.0',    # Новая версия
    treatment_traffic_pct=10,      # 10% на новую
    duration_days=7,
    min_requests=1000,
    max_error_rate_increase=0.05,  # +5% errors = fail
    min_confidence_improvement=0.02 # +2% confidence = success
)

ab_test_manager.start_test(test_config)

# 2. В diagnosis service: получить вариант для пользователя
variant = ab_test_manager.assign_variant('gnn', user_id='user123')
print(variant)
# {'name': 'control', 'version': 'v2.1.0', 'model': <ModelVersion>}

# 3. Записывать результаты
ab_test_manager.record_result(
    test_name='gnn_v2.2_test',
    variant='control',  # or 'treatment'
    inference_time_ms=124.5,
    confidence=0.87,
    error=False
)

# 4. Оценить результаты (через 7 дней)
evaluation = ab_test_manager.evaluate_test('gnn_v2.2_test')
print(evaluation)
# {
#   'decision': 'promote',  # or 'rollback' or 'continue'
#   'metrics': {...},
#   'recommendation': 'Confidence улучшился на 2.5%. Промоутим treatment.'
# }

# 5. Завершить тест
if evaluation['decision'] == 'promote':
    ab_test_manager.finalize_test('gnn_v2.2_test', 'promote')
    # v2.2.0 становится champion
```

### Success Criteria

**Promote** (treatment → production):
- Error rate не увеличился больше `max_error_rate_increase`
- Confidence улучшился на `min_confidence_improvement`
- Набрано `min_requests` запросов

**Rollback** (откат на control):
- Error rate увеличился > `max_error_rate_increase`

---

## 3️⃣ Drift Detection

### Пример

```python
from mlops import get_drift_detector

# Получить detector для GNN v2.1.0
detector = get_drift_detector('gnn', 'v2.1.0')

# 1. Заполнить reference distribution (первые 10k samples)
for i in range(10000):
    prediction = ...  # anomaly_score from model
    confidence = ...  # model confidence
    detector.add_reference_sample(prediction, confidence)

# 2. Мониторинг production
for sample in production_data:
    prediction = model.predict(sample)
    detector.add_production_sample(prediction['anomaly_score'], prediction['confidence'])

# 3. Проверить drift
drift_score = detector.detect_drift()
if drift_score > 0.3:
    print("⚠️ Drift detected!")

# 4. Получить отчёт
report = detector.get_drift_report()
print(report)
# {
#   'status': 'active',
#   'current_drift_score': 0.42,
#   'avg_drift_7d': 0.35,
#   'alert_triggered': True,
#   'recommendation': 'Критический drift! Требуется срочное переобучение.'
# }
```

### Drift Methods

1. **Statistical Drift**: Kolmogorov-Smirnov test на распределение predictions
2. **Performance Drift**: Снижение confidence
3. **Distribution Shift**: Изменение mean/variance predictions

**Combined score** = 0.4 × statistical + 0.3 × performance + 0.3 × distribution

### Thresholds

- `drift_score < 0.1` → 🟢 Норма
- `0.1 < drift_score < 0.3` → 🟡 Мониторим
- `0.3 < drift_score < 0.5` → 🟠 Рекомендуем переобучение
- `drift_score > 0.5` → 🔴 Критично! Срочное переобучение

---

## 📈 Prometheus Metrics

All drift scores экспортируются в Prometheus:

```prometheus
# Drift score per model
model_drift_score{model_type="gnn", model_version="v2.1.0"}
```

**Grafana dashboard:**

```promql
# Alert when drift > 0.3
model_drift_score > 0.3
```

---

## 🚀 Интеграция в Diagnosis Service

### `main.py`

```python
from mlops import model_registry, ab_test_manager, get_drift_detector
from monitoring.metrics import record_gnn_metrics, record_rag_metrics

@app.post("/diagnosis/create")
async def create_diagnosis(equipment_id: str, user_id: str):
    # A/B testing: выбрать версию GNN
    gnn_variant = ab_test_manager.assign_variant('gnn', user_id)
    gnn_version = gnn_variant['version']
    
    # GNN inference
    gnn_result = await gnn_service.predict(
        equipment_id=equipment_id,
        model_version=gnn_version
    )
    
    # Записать метрики
    record_gnn_metrics(
        model_version=gnn_version,
        inference_time_ms=gnn_result['inference_time_ms'],
        anomaly_score=gnn_result['anomaly_score'],
        confidence=gnn_result['confidence']
    )
    
    # Drift detection
    drift_detector = get_drift_detector('gnn', gnn_version)
    drift_detector.add_production_sample(
        gnn_result['anomaly_score'],
        gnn_result['confidence']
    )
    
    # A/B test: записать результат
    if 'gnn_v2.2_test' in ab_test_manager._active_tests:
        ab_test_manager.record_result(
            test_name='gnn_v2.2_test',
            variant=gnn_variant['name'],
            inference_time_ms=gnn_result['inference_time_ms'],
            confidence=gnn_result['confidence'],
            error=False
        )
    
    return gnn_result
```

---

## 🛠️ Утилиты

### Просмотр всех версий

```python
from mlops import model_registry

# GNN versions
for v in model_registry.list_versions('gnn'):
    print(f"{v.version}: champion={v.is_champion}, traffic={v.traffic_percentage}%")
```

### Промоут версии вручную

```python
model_registry.promote_to_champion('gnn', 'v2.2.0')
```

### Проверить drift всех моделей

```python
from mlops.drift_detector import _drift_detectors

for key, detector in _drift_detectors.items():
    report = detector.get_drift_report()
    print(f"{key}: {report['current_drift_score']:.3f}")
```

---

## 📊 Next Steps

1. **Persistent storage**: Перевести `model_registry` на PostgreSQL/Redis
2. **Auto-retraining**: Trigger retraining при drift > 0.5
3. **Rollback automation**: Автооткат при error_rate spike
4. **Model registry UI**: Dashboard для управления версиями
