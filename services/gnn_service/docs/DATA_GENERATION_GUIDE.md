# 📊 Data Generation Guide (Phase 2)

Полное руководство по генерации данных для обучения GNN модели.

---

## 🚀 Quick Start

### Базовая генерация (3 строки):

```python
from src.data_generation import HydraulicScenarioGenerator

generator = HydraulicScenarioGenerator()
all_data = generator.generate_all(save_dir="data/generated")
```

**Результат:**
```
data/generated/
├── parallel_operations_700graphs.pt       # 700 графов (7 узлов)
├── sequential_cascade_400graphs.pt        # 400 графов (10 узлов)
└── temporal_sequences_100x10.pt           # 100 последовательностей × 10 шагов
```

---

## 📦 Phase 2 Output Format

### Каждый граф содержит:

```python
graph = all_data['parallel']['graphs'][0]

# Node features
graph.x                    # [N, 34] - фичи узлов
graph.edge_index           # [2, E] - граф связей
graph.edge_attr            # [E, 8] - фичи рёбер (модель проецирует в 14D)

# ===== Graph-level targets (4 задачи) =====
graph.y_graph_health       # [1] ∈ [0,1] - общее здоровье системы
graph.y_graph_degradation  # [1] ∈ [0,1] - скорость деградации
graph.y_graph_anomaly      # [9] ∈ {0,1}^9 - 9 типов аномалий (multi-label)
graph.y_graph_rul          # [1] ∈ [0,∞) - остаточный ресурс (часы)

# ===== Component-level targets (2 задачи) =====
graph.y_component_health   # [N] ∈ [0,1] - здоровье каждого узла
graph.y_component_anomaly  # [N,9] ∈ {0,1}^9 - аномалии каждого узла

# Batch tensor (для DataLoader)
graph.batch                # [N] - индексы батча
```

### 9 классов аномалий:

```python
ANOMALY_CLASSES = [
    'overload',           # 0: Перегрузка
    'pressure_spike',     # 1: Скачок давления
    'cavitation',         # 2: Кавитация
    'contamination',      # 3: Загрязнение
    'leakage',            # 4: Утечка
    'valve_stuck',        # 5: Заклинивание клапана
    'pump_degradation',   # 6: Износ насоса
    'thermal_runaway',    # 7: Перегрев
    'vibration_anomaly',  # 8: Аномальная вибрация
]
```

---

## 🎯 Типы сценариев

### 1️⃣ **Parallel Operations** (700 графов)

**Топология:** 7 узлов (2 насоса → load-sensing клапан → 4 актуатора)

**Симулирует:**
- Одновременная работа boom + swing
- Динамическое распределение потока
- Load-sensing приоритизация

**Генерация:**
```python
parallel_data = generator.generate_parallel_scenarios()
print(len(parallel_data['graphs']))  # 700
```

**Основные аномалии:**
- `overload` (40% сценариев)
- `pressure_spike`
- `cavitation`

---

### 2️⃣ **Sequential Cascades** (400 графов)

**Топология:** 10 узлов (насос → relief → distributor → section relief → motor → shock/makeup)

**Симулирует:**
- Каскад предохранительных клапанов
- Overload → main relief → section relief цепочка
- Последовательные отказы

**Генерация:**
```python
sequential_data = generator.generate_sequential_scenarios()
print(len(sequential_data['graphs']))  # 400
```

**Основные аномалии:**
- `sequential_cascade` (40% сценариев)
- `valve_stuck`
- `pressure_spike`

---

### 3️⃣ **Temporal Sequences** (100 × 10 timesteps)

**Топологии:** Mixed (3, 7, 10 узлов)

**Симулирует:**
- Постепенная деградация (healthy → degraded → failed)
- Временная эволюция системы
- Progressive failure patterns

**Генерация:**
```python
temporal_data = generator.generate_temporal_sequences()

for seq in temporal_data['sequences']:
    print(len(seq))  # 10 timesteps
    print(seq[0].y_graph_health)  # Здоровая система
    print(seq[-1].y_graph_health)  # Деградировавшая система
```

**Progression pattern:**
```
Timestep 0-2:   Normal (health > 0.8)
Timestep 3-6:   Degrading (health 0.5-0.8)
Timestep 7-9:   Critical (health < 0.5, anomalies появляются)
```

---

## ⚙️ Advanced Configuration

### Custom Generator Config:

```python
from src.data_generation import GeneratorConfig, HydraulicScenarioGenerator

config = GeneratorConfig(
    # Dataset sizes
    num_parallel_graphs=1000,        # Увеличить parallel сценарии
    num_sequential_graphs=500,
    num_temporal_sequences=200,
    temporal_sequence_length=15,     # Длиннее последовательности
    
    # Noise & variation
    noise_level=0.1,                 # 10% шум (было 5%)
    degradation_probability=0.5,     # Больше деградации
    anomaly_probability=0.6,         # Больше аномалий
    
    # Physical parameters
    nominal_pressure_range=(150.0, 400.0),  # Wider range
    nominal_flow_range=(30.0, 350.0),
    temperature_range=(30.0, 100.0),
    
    # RUL simulation
    rul_healthy_range=(800.0, 1500.0),   # Более долгоживущие системы
    rul_degraded_range=(5.0, 800.0),
    
    # Validation
    validate_physics=True,            # Включить валидацию
    strict_validation=False,          # Warnings only (не падать)
    
    # Reproducibility
    random_seed=123,                  # Custom seed
)

generator = HydraulicScenarioGenerator(config)
data = generator.generate_all(save_dir="data/custom")
```

---

## 🔧 Integration with Training

### 1. Генерация данных:

```python
# generate_data.py
from src.data_generation import HydraulicScenarioGenerator

generator = HydraulicScenarioGenerator()
all_data = generator.generate_all(save_dir="data/generated")

print(f"Generated {all_data['parallel']['num_graphs']} parallel graphs")
print(f"Generated {all_data['sequential']['num_graphs']} sequential graphs")
print(f"Generated {all_data['temporal']['num_sequences']} temporal sequences")
```

### 2. Загрузка в Dataset:

```python
from src.data.dataset import TemporalGraphDataset
from src.data.feature_config import FeatureConfig

feature_config = FeatureConfig(edge_in_dim=14)

dataset = TemporalGraphDataset(
    data_path="data/generated/parallel_operations_700graphs.pt",
    feature_config=feature_config,
    split="train",
)

print(len(dataset))  # 700
print(dataset[0].y_graph_health)  # [1]
print(dataset[0].y_component_anomaly.shape)  # [N, 9]
```

### 3. Train/Val/Test Split:

```python
import torch
from torch.utils.data import random_split

# Load generated data
parallel_data = torch.load("data/generated/parallel_operations_700graphs.pt")
graphs = parallel_data['graphs']

# Split: 70% train, 15% val, 15% test
train_size = int(0.7 * len(graphs))
val_size = int(0.15 * len(graphs))
test_size = len(graphs) - train_size - val_size

train_graphs, val_graphs, test_graphs = random_split(
    graphs, [train_size, val_size, test_size]
)

# Save splits
torch.save({'graphs': list(train_graphs)}, "data/train.pt")
torch.save({'graphs': list(val_graphs)}, "data/val.pt")
torch.save({'graphs': list(test_graphs)}, "data/test.pt")
```

### 4. Training with DataLoader:

```python
from torch_geometric.loader import DataLoader

train_dataset = TemporalGraphDataset(
    data_path="data/train.pt",
    feature_config=feature_config,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for batch in train_loader:
    # batch.y_graph_health: [32, 1]
    # batch.y_component_anomaly: [N_total, 9]
    pass
```

---

## 🧪 Validation & Physics Checks

### Включить строгую валидацию:

```python
config = GeneratorConfig(
    validate_physics=True,
    strict_validation=True,  # Raise errors on violations
)

generator = HydraulicScenarioGenerator(config)

try:
    data = generator.generate_all()
except ValueError as e:
    print(f"Physics violation: {e}")
```

### Проверяемые условия:

1. **Pressure consistency:**
   - `P_source > P_target` (давление падает)
   - No negative pressures

2. **Flow conservation:**
   - ∑ flow_in ≈ ∑ flow_out (для каждого узла)

3. **Temperature bounds:**
   - 30°C ≤ T ≤ 100°C

4. **Edge features validity:**
   - pipe_diameter > 0
   - pipe_length > 0
   - pressure_drop ≥ 0

### Warnings (strict_validation=False):

```python
# Генератор выведет предупреждения, но не упадёт:
"""
WARNING: Node 3 pressure (45.2 bar) below nominal (200 bar)
WARNING: Edge 5 flow velocity (15.3 m/s) exceeds typical range
"""
```

---

## 📊 Dataset Statistics

### Получить статистику:

```python
from src.data.dataset import TemporalGraphDataset

dataset = TemporalGraphDataset(
    data_path="data/generated/parallel_operations_700graphs.pt",
    feature_config=feature_config,
)

stats = dataset.get_statistics()
print(stats)
```

**Output:**
```python
{
    'dataset_size': 700,
    'split': 'train',
    'avg_num_nodes': 7.0,
    'min_num_nodes': 7,
    'max_num_nodes': 7,
    'avg_num_edges': 12.0,
    'node_features': 34,
    'edge_feature_dims': [8],           # Loaded dimension
    'edge_in_dim_configured': 14,       # Model expects 14D
    'sample_size': 10,
    
    # Target statistics
    'avg_graph_health': 0.73,
    'avg_graph_rul': 342.5,
    'avg_component_health': 0.68,
    'anomaly_prevalence': 0.42,         # 42% graphs have anomalies
}
```

---

## 🛠️ Troubleshooting

### ❌ **Problem:** `KeyError: 'graphs'`

**Reason:** Старый формат данных (до Phase 2)

**Fix:**
```python
# Regenerate data with Phase 2 generator
generator = HydraulicScenarioGenerator()
data = generator.generate_all(save_dir="data/generated")
```

---

### ❌ **Problem:** `RuntimeError: edge_attr dimension mismatch`

**Reason:** Generator создаёт 8D edge features, модель ожидает 14D

**Fix:**
```python
# This is EXPECTED behavior!
# Generator: 8D static features
# Model: Projects to 14D (8D static + 6D dynamic)

# Just ensure FeatureConfig matches:
config = FeatureConfig(edge_in_dim=14)  # Model expects 14D
```

---

### ❌ **Problem:** `ValueError: Physics validation failed`

**Reason:** Generated graph violates physical constraints

**Fix 1 (disable validation):**
```python
config = GeneratorConfig(validate_physics=False)
```

**Fix 2 (warnings only):**
```python
config = GeneratorConfig(
    validate_physics=True,
    strict_validation=False,  # Don't raise errors
)
```

---

### ❌ **Problem:** Memory issues with large datasets

**Reason:** Генерация 1000+ графов в памяти

**Fix (generate in chunks):**
```python
from pathlib import Path

save_dir = Path("data/generated")
save_dir.mkdir(parents=True, exist_ok=True)

# Generate parallel (700 graphs)
parallel_data = generator.generate_parallel_scenarios()
torch.save(parallel_data, save_dir / "parallel.pt")
del parallel_data  # Free memory

# Generate sequential (400 graphs)
sequential_data = generator.generate_sequential_scenarios()
torch.save(sequential_data, save_dir / "sequential.pt")
del sequential_data

# Generate temporal (100 sequences)
temporal_data = generator.generate_temporal_sequences()
torch.save(temporal_data, save_dir / "temporal.pt")
```

---

## 📚 See Also

- **[Model Architecture](MODEL_ARCHITECTURE.md)** - UniversalTemporalGNNv2 details
- **[Training Guide](TRAINING_GUIDE.md)** - Full training pipeline
- **[Feature Engineering](FEATURE_ENGINEERING.md)** - Node/edge feature specs
- **[API Reference](API_REFERENCE.md)** - Complete API docs

---

## 🎯 Next Steps

1. ✅ Сгенерировать данные: `generator.generate_all()`
2. ✅ Проверить статистику: `dataset.get_statistics()`
3. ✅ Split на train/val/test
4. ✅ Начать обучение: `src/training/train.py`

**Готов к обучению модели!** 🚀
