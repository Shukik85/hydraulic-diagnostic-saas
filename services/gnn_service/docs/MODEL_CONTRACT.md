# UniversalTemporalGNN Model Contract

**Version:** 2.0.1  
**Updated:** 2025-12-04  
**Status:** Production Ready

---

## Overview

`UniversalTemporalGNN` — универсальная пространственно-временная графовая нейронная сеть для диагностики гидравлических систем. Модель **инвариантна** к размеру графа, числу рёбер и батч-размеру, что позволяет использовать её для систем с различным числом компонентов и конфигураций.

### Ключевые свойства универсальности

1. **Инвариантность к числу узлов (N):**  
   Модель корректно работает с графами от 1 до N узлов без изменения архитектуры.

2. **Инвариантность к числу рёбер (E):**  
   Количество соединений между компонентами может быть произвольным.

3. **Инвариантность к батч-размеру (B):**  
   Возможна обработка одного графа или батча из нескольких графов разного размера.

4. **Гибкость по размерности edge-фич (edge_in_dim):**  
   С версии 2.0.1 модель поддерживает произвольную размерность признаков рёбер через слой `edge_projection`.

---

## Входной контракт

### Обязательные входы

```python
def forward(
    self,
    x: torch.Tensor,           # Node features [N_total, F_node]
    edge_index: torch.Tensor,  # Edge connectivity [2, E_total]
    edge_attr: torch.Tensor | None,  # Edge features [E_total, F_edge] or None
    batch: torch.Tensor,       # Batch assignment [N_total]
) -> dict[str, dict[str, torch.Tensor]]:
```

### Детальное описание

#### 1. `x: torch.Tensor` — Node Features
- **Shape:** `[N_total, F_node]`
- **Dtype:** `torch.float32`
- **Описание:** Признаки узлов (компонентов)
  - `N_total` — общее число узлов во всех графах батча
  - `F_node` — размерность признаков узла (задаётся параметром `in_channels` при инициализации)
- **Пример:** `x.shape = [150, 34]` для батча из 3 графов с 50 узлами каждый

#### 2. `edge_index: torch.Tensor` — Edge Connectivity
- **Shape:** `[2, E_total]`
- **Dtype:** `torch.long`
- **Описание:** Индексы соединений в формате COO (PyTorch Geometric)
  - `edge_index[0]` — source nodes
  - `edge_index[1]` — target nodes
  - `E_total` — общее число рёбер во всех графах батча
- **Пример:** `edge_index.shape = [2, 300]` для 300 соединений

#### 3. `edge_attr: torch.Tensor | None` — Edge Features
- **Shape:** `[E_total, F_edge]` или `None`
- **Dtype:** `torch.float32`
- **Описание:** Признаки рёбер (соединений)
  - `E_total` — число рёбер (совпадает с `edge_index.shape[1]`)
  - `F_edge` — размерность признаков ребра (задаётся параметром `edge_in_dim`)
  - **Если `None`:** модель работает без признаков рёбер (используется только топология)
- **Пример:** `edge_attr.shape = [300, 14]` для 14D признаков на каждое ребро

#### 4. `batch: torch.Tensor` — Batch Assignment
- **Shape:** `[N_total]`
- **Dtype:** `torch.long`
- **Описание:** Индексы графов для каждого узла
  - `batch[i] = j` означает, что узел `i` принадлежит графу `j` в батче
  - Для одного графа: `batch = torch.zeros(N, dtype=torch.long)`
- **Пример:** `batch = [0,0,0, ..., 1,1,1, ..., 2,2,2]` для батча из 3 графов

---

## Выходной контракт

```python
{
    "component": {
        "health": torch.Tensor,    # [N_total, 1] ∈ [0, 1]
        "anomaly": torch.Tensor    # [N_total, 9] logits
    },
    "graph": {
        "health": torch.Tensor,       # [B, 1] ∈ [0, 1]
        "degradation": torch.Tensor,  # [B, 1] ∈ [0, 1]
        "anomaly": torch.Tensor,      # [B, 9] logits
        "rul": torch.Tensor          # [B, 1] ∈ [0, ∞)
    }
}
```

### Детальное описание выходов

#### Component-Level (Per-Node)

1. **`component.health`:** `[N_total, 1]`  
   - Здоровье каждого компонента (0 = неисправен, 1 = исправен)
   - Активация: Sigmoid ∈ [0, 1]

2. **`component.anomaly`:** `[N_total, 9]`  
   - Логиты для 9 типов аномалий на уровне компонента
   - Используйте `torch.softmax` или `torch.sigmoid` для вероятностей

#### Graph-Level (Per-Equipment)

1. **`graph.health`:** `[B, 1]`  
   - Общее здоровье системы (0 = критично, 1 = отлично)
   - Активация: Sigmoid ∈ [0, 1]

2. **`graph.degradation`:** `[B, 1]`  
   - Степень деградации системы (0 = нет, 1 = полная)
   - Активация: Sigmoid ∈ [0, 1]

3. **`graph.anomaly`:** `[B, 9]`  
   - Логиты для 9 типов аномалий на уровне системы
   - Типы: [cooler, valve, pump, accumulator, stable, small_lag, severe_lag, close_to_total_failure, full_degradation]

4. **`graph.rul`:** `[B, 1]`  
   - Remaining Useful Life в часах (0 = требуется немедленная замена, ∞ = новое)
   - Активация: Softplus ∈ [0, ∞)

---

## Примеры использования

### Пример 1: Один граф (инференс)

```python
import torch
from src.models.universal_temporal_gnn import UniversalTemporalGNN

# Инициализация модели
model = UniversalTemporalGNN(
    in_channels=34,      # 34 признака на узел
    hidden_channels=128,
    edge_in_dim=14,      # 14 признаков на ребро
    num_heads=8,
    num_gat_layers=3
)

# Один граф: 50 узлов, 80 рёбер
x = torch.randn(50, 34)              # Node features
edge_index = torch.randint(0, 50, (2, 80))  # Edge connectivity
edge_attr = torch.randn(80, 14)      # Edge features
batch = torch.zeros(50, dtype=torch.long)   # Все узлы в графе 0

# Forward pass
output = model(x, edge_index, edge_attr, batch)

print(output['graph']['health'].shape)      # [1, 1]
print(output['component']['health'].shape)  # [50, 1]
```

### Пример 2: Батч из разных графов

```python
from torch_geometric.data import Data, Batch

# Граф 1: 30 узлов, 45 рёбер
graph1 = Data(
    x=torch.randn(30, 34),
    edge_index=torch.randint(0, 30, (2, 45)),
    edge_attr=torch.randn(45, 14)
)

# Граф 2: 70 узлов, 120 рёбер
graph2 = Data(
    x=torch.randn(70, 34),
    edge_index=torch.randint(0, 70, (2, 120)),
    edge_attr=torch.randn(120, 14)
)

# Граф 3: 25 узлов, 30 рёбер
graph3 = Data(
    x=torch.randn(25, 34),
    edge_index=torch.randint(0, 25, (2, 30)),
    edge_attr=torch.randn(30, 14)
)

# Создание батча
batch_data = Batch.from_data_list([graph1, graph2, graph3])

# Forward pass
output = model(
    x=batch_data.x,              # [125, 34] (30+70+25)
    edge_index=batch_data.edge_index,  # [2, 195] (45+120+30)
    edge_attr=batch_data.edge_attr,    # [195, 14]
    batch=batch_data.batch       # [125] (индексы 0,1,2)
)

print(output['graph']['health'].shape)      # [3, 1] - 3 графа
print(output['component']['health'].shape)  # [125, 1] - 125 узлов
```

### Пример 3: Разная размерность edge-фич

```python
# Модель с 8D edge features (backward compatible)
model_8d = UniversalTemporalGNN(
    in_channels=34,
    hidden_channels=128,
    edge_in_dim=8,  # Старый формат
    num_heads=8
)

# Модель с 20D edge features (новая конфигурация)
model_20d = UniversalTemporalGNN(
    in_channels=34,
    hidden_channels=128,
    edge_in_dim=20,  # Расширенные признаки
    num_heads=8
)

# Оба варианта корректно работают
edge_attr_8d = torch.randn(100, 8)
edge_attr_20d = torch.randn(100, 20)

out_8d = model_8d(x, edge_index, edge_attr_8d, batch)
out_20d = model_20d(x, edge_index, edge_attr_20d, batch)
```

### Пример 4: Без edge-фич

```python
# Модель может работать только с топологией (без edge_attr)
model = UniversalTemporalGNN(
    in_channels=34,
    hidden_channels=128,
    edge_in_dim=8,  # Будет игнорироваться если edge_attr=None
    num_heads=8
)

# Forward без edge features
output = model(
    x=x,
    edge_index=edge_index,
    edge_attr=None,  # Без признаков рёбер
    batch=batch
)

# Модель использует только структуру графа
```

---

## Архитектура обработки edge-фич

### Edge Projection Layer (v2.0.1+)

```python
# Инициализация в __init__
self.edge_projection = nn.Sequential(
    nn.Linear(edge_in_dim, edge_hidden_dim),  # edge_hidden_dim = hidden_channels // num_heads
    nn.LayerNorm(edge_hidden_dim),
    nn.ReLU(),
    nn.Dropout(dropout),
)

# Использование в forward
if edge_attr is not None:
    edge_emb = self.edge_projection(edge_attr)  # [E, edge_in_dim] → [E, edge_hidden_dim]
else:
    edge_emb = None

# Передача в GAT
h_new = gat_layer(h, edge_index, edge_emb)
```

**Зачем нужен edge_projection:**
1. Позволяет использовать произвольную размерность входных edge-фич
2. Приводит edge-фичи к размерности, совместимой с GATv2
3. Добавляет нелинейное преобразование для лучшего обучения
4. Сохраняет обратную совместимость (edge_in_dim=8 по умолчанию)

---

## Backward Compatibility

### Миграция с версии <2.0.1

**Старый код (v2.0.0 и ранее):**
```python
model = UniversalTemporalGNN(
    in_channels=34,
    hidden_channels=128,
    num_heads=8
)
# edge_attr всегда должен быть [E, 8]
```

**Новый код (v2.0.1+):**
```python
model = UniversalTemporalGNN(
    in_channels=34,
    hidden_channels=128,
    edge_in_dim=8,  # Явно указываем (необязательно, default=8)
    num_heads=8
)
# edge_attr может быть [E, 8] или [E, любая_размерность]
```

### Загрузка старых чекпоинтов

```python
import torch

# Загрузка старого чекпоинта (до v2.0.1)
checkpoint = torch.load('model_v2.0.0.ckpt')

# Инициализация модели с edge_in_dim=8 (совместимость)
model = UniversalTemporalGNN(
    in_channels=34,
    hidden_channels=128,
    edge_in_dim=8,  # Важно: должно совпадать с обучением
    num_heads=8
)

# Загрузка весов
model.load_state_dict(checkpoint['state_dict'])

# Модель готова к инференсу
```

**Примечание:** Старые чекпоинты **не содержат** веса `edge_projection`, поэтому при загрузке будет предупреждение о несовпадении ключей. Это нормально — нужно либо:
1. Дообучить модель с новым слоем `edge_projection`
2. Использовать модель без edge_projection (не рекомендуется)

---

## Валидация входов

### Runtime Checks (рекомендуется)

```python
def validate_inputs(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor | None,
    batch: torch.Tensor,
    expected_in_channels: int,
    expected_edge_in_dim: int
) -> None:
    """Validate model inputs before forward pass."""
    
    # 1. Check node features
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}D"
    assert x.size(1) == expected_in_channels, \
        f"x.size(1)={x.size(1)}, expected {expected_in_channels}"
    
    # 2. Check edge connectivity
    assert edge_index.dim() == 2, f"edge_index must be 2D, got {edge_index.dim()}D"
    assert edge_index.size(0) == 2, \
        f"edge_index.size(0) must be 2, got {edge_index.size(0)}"
    
    # 3. Check edge features (if provided)
    if edge_attr is not None:
        assert edge_attr.dim() == 2, f"edge_attr must be 2D, got {edge_attr.dim()}D"
        assert edge_attr.size(0) == edge_index.size(1), \
            f"edge_attr.size(0)={edge_attr.size(0)} != edge_index.size(1)={edge_index.size(1)}"
        assert edge_attr.size(1) == expected_edge_in_dim, \
            f"edge_attr.size(1)={edge_attr.size(1)}, expected {expected_edge_in_dim}"
    
    # 4. Check batch assignment
    assert batch.dim() == 1, f"batch must be 1D, got {batch.dim()}D"
    assert batch.size(0) == x.size(0), \
        f"batch.size(0)={batch.size(0)} != x.size(0)={x.size(0)}"
    
    # 5. Check edge indices are in range
    assert edge_index.min() >= 0, "edge_index contains negative indices"
    assert edge_index.max() < x.size(0), \
        f"edge_index.max()={edge_index.max()} >= x.size(0)={x.size(0)}"

# Usage
validate_inputs(x, edge_index, edge_attr, batch, 
                expected_in_channels=34, expected_edge_in_dim=14)
output = model(x, edge_index, edge_attr, batch)
```

---

## FAQ

### Q: Что делать, если в разных графах разное число узлов?
**A:** Это штатный режим работы. Используйте `torch_geometric.data.Batch` для создания батча из графов разного размера.

### Q: Можно ли использовать модель без edge-фич?
**A:** Да, передайте `edge_attr=None`. Модель будет использовать только топологию графа.

### Q: Как изменить размерность edge-фич без переобучения?
**A:** Нельзя. Размерность `edge_in_dim` жёстко связана с весами `edge_projection`. Нужно переобучить модель или дообучить новый слой.

### Q: Влияет ли порядок узлов на результат?
**A:** Нет. GNN инвариантна к перестановкам узлов благодаря использованию `edge_index` и graph pooling.

### Q: Как обрабатываются self-loops?
**A:** GATv2Conv автоматически добавляет self-loops (`add_self_loops=True`). Вручную добавлять не нужно.

### Q: Что если в батче только один граф?
**A:** Модель корректно работает. `batch = torch.zeros(N, dtype=torch.long)` для N узлов.

---

## Рекомендации для Production

1. **Валидация входов:**  
   Всегда проверяйте размерности и типы перед вызовом `model.forward()`.

2. **Батчинг:**  
   Используйте батчи графов для повышения throughput (особенно на GPU).

3. **torch.compile:**  
   Включите `use_compile=True` для ускорения инференса (PyTorch 2.8+).

4. **GPU memory pinning:**  
   Используйте `pin_memory=True` в DataLoader для быстрой передачи на GPU.

5. **edge_in_dim:**  
   Зафиксируйте размерность edge-фич в конфиге и не меняйте между обучением и инференсом.

---

## См. также

- [API_DOCS.md](API_DOCS.md) — API reference
- [STRUCTURE.md](../STRUCTURE.md) — Архитектура сервиса
- [CHANGELOG.md](../CHANGELOG.md) — История изменений
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)

---

**Последнее обновление:** 2025-12-04 23:45 MSK  
**Версия контракта:** 2.0.1  
**Автор:** GNN Service Team