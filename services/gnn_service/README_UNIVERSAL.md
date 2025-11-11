# Universal GNN Service

## 🎯 Назначение

Production-ready сервис для универсальной диагностики гидравлических систем на основе Graph Neural Networks. Динамически адаптируется под любую топологию системы (пресс, экскаватор, кран, пользовательские конфигурации).

## 🏗️ Архитектура

```
services/gnn_service/
├── model_universal.py           # Универсальная GNN модель
├── graph_builder.py             # Построение графов из metadata
├── inference_service.py         # FastAPI REST API
├── openapi.yaml                 # API спецификация
└── README_UNIVERSAL.md          # Эта документация
```

## 🚀 Компоненты

### 1. Universal GNN Model (`model_universal.py`)

**Ключевые возможности:**
- Динамическая адаптация под любое количество компонентов
- GAT (Graph Attention) для weighted connections
- Multi-head attention (4 heads)
- Dropout для регуляризации
- Поддержка разнородных сенсоров

**Архитектура:**
```
Input: Node features (N × F)  # N компонентов, F фич на каждый
  ↓
GAT Layer 1 (F → 64, 4 heads)
  ↓
ReLU + Dropout(0.3)
  ↓
GAT Layer 2 (64 → 32, 4 heads)
  ↓
ReLU + Dropout(0.3)
  ↓
Linear (32 → 1)  # Anomaly score per component
  ↓
Output: Логиты (N × 1)  # Sigmoid → [0,1] вероятность дефекта
```

### 2. Graph Builder (`graph_builder.py`)

**Функции:**
- `build_node_features(data_df, metadata)` — создание feature vectors для каждого компонента
- `adjacency_to_edge_index(adjacency_matrix)` — конвертация матрицы смежности в edge_index (PyG format)

**Пример:**
```python
# Metadata
{
  "components": [
    {"id": "pump", "sensors": ["pressure", "flow", "temp"]},
    {"id": "valve", "sensors": ["position", "pressure"]}
  ],
  "adjacency_matrix": [[0, 1], [1, 0]]  # pump ↔ valve
}

# Data (5-минутный срез из TimescaleDB)
timestamp, pump_pressure, pump_flow, pump_temp, valve_position, valve_pressure
...

# Graph Builder → PyTorch Geometric Data
node_features = [[p_mean, f_mean, t_mean],  # pump
                 [pos_mean, p_mean]]        # valve
edge_index = [[0, 1], [1, 0]]  # bidirectional
```

### 3. Inference Service (`inference_service.py`)

**FastAPI endpoints:**

#### `POST /gnn/infer`
Универсальный inference для любой системы.

**Query Parameters:**
- `user_id` (str) — ID пользователя
- `system_id` (str) — ID системы (пресс, экскаватор и т.д.)

**Response:**
```json
{
  "system_id": "press_01",
  "anomaly_scores": {
    "pump": 0.05,
    "valve_main": 0.87,  // ⚠️ аномалия!
    "cylinder": 0.12
  },
  "n_components": 3
}
```

#### `GET /gnn/health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## 📊 Workflow

```
1. User/System metadata → equipment_metadata.json
2. TimescaleDB → последние 5 мин sensor data
3. Graph Builder → PyG Data (node_features, edge_index)
4. Universal GNN → inference
5. FastAPI → JSON response с аномалиями
```

## 🔧 Установка и запуск

### Зависимости
```bash
pip install torch torch-geometric fastapi uvicorn pandas
```

### Запуск сервиса
```bash
cd services/gnn_service
uvicorn inference_service:app --host 0.0.0.0 --port 8001
```

### Тестирование
```bash
curl -X POST "http://localhost:8001/gnn/infer?user_id=user_123&system_id=press_01"
```

## 📝 Интеграция с Backend

### Django DRF → GNN Service
```python
# backend/views.py
import requests

def get_anomalies(user_id, system_id):
    response = requests.post(
        "http://gnn_service:8001/gnn/infer",
        params={"user_id": user_id, "system_id": system_id}
    )
    return response.json()
```

### Nuxt Frontend → Backend → GNN
```typescript
// composables/useGnn.ts
export const useGnn = () => {
  const runInference = async (userId: string, systemId: string) => {
    const { data } = await useFetch('/api/gnn/infer', {
      params: { user_id: userId, system_id: systemId }
    });
    return data.value;
  };
  return { runInference };
};
```

## 🧪 Тестирование

### Unit тесты
```python
# tests/test_model_universal.py
def test_universal_gnn():
    metadata = {...}
    model = UniversalHydraulicGNN(metadata)
    x = torch.randn(5, 10)  # 5 компонентов, 10 фич
    edge_index = torch.tensor([[0,1,2], [1,2,3]])
    out = model(x, edge_index)
    assert out.shape == (5, 1)
```

### Integration тест
```bash
pytest tests/test_inference_service.py
```

## 🚀 Production Checklist

- ✅ Multi-tenant изоляция (user_id/system_id)
- ✅ Динамическая топология (любые системы)
- ✅ Health check endpoint
- ✅ OpenAPI документация
- ✅ Logging (structlog)
- ⏳ Model versioning (MLflow)
- ⏳ A/B testing для новых моделей
- ⏳ Rate limiting
- ⏳ Monitoring (Prometheus metrics)

## 📈 Roadmap

### Phase 1 (текущая) ✅
- Universal GNN модель
- Graph builder
- REST API

### Phase 2
- Temporal GNN (с учётом истории)
- Explainability (GNNExplainer)
- Model registry (MLflow)

### Phase 3
- Real-time inference (WebSocket)
- Federated learning (multi-tenant обучение)
- AutoML для гиперпараметров

## 📞 Support

Вопросы: shukik85@ya.ru
