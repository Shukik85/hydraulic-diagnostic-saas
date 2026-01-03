# 🏗️ Architecture Redesign: Production Data Pipeline

**Status**: Architecture Design Document  
**Date**: January 3, 2026  
**Context**: Confusion cleared - starting from honest baseline

---

## Executive Summary

Ты прав. Нужна **полная переработка архитектуры данных**.

**Текущее состояние**:
- ✅ Реальные данные есть (UC Irvine dataset, ~450 MB)
- ❌ Архитектура обработки - нет
- ❌ Подготовка к обучению - нет
- ❌ Синтетическое расширение - нет
- ❌ Семисинтетические признаки - нет

**Что нужно**:
1. **Phase 3A**: Подготовка реальных данных (1-2 недели)
2. **Phase 3B**: Физически обоснованная генерация синтетики (2-3 недели)
3. **Phase 3C**: Семисинтетические признаки с топологией (1 неделя)
4. **Phase 3D**: Полный pipeline train/val/test (1 неделя)

---

## Phase 1: Текущие данные (UC Irvine)

### Структура данных

```
services/gnn_service/data/raw_real_dataset/
├── PS*.txt        # Pressure sensor (6 файлов) - ~380 MB
├── FS*.txt        # Flow rate sensor (2 файла) - ~16 MB
├── EPS1.txt       # Electric power sensor - ~87 MB
├── CE.txt         # Contaminant level - ~911 KB
├── CP.txt         # Cooler Power - ~779 KB
├── SE.txt         # Specific Energy - ~825 KB
├── TS*.txt        # Temperature sensor (4 файла) - ~3.6 MB
├── VS1.txt        # Vibration sensor - ~778 KB
├── profile.txt    # Профиль системы
├── documentation.txt # Метаданные
└── description.txt   # Описание
```

**Total**: ~450 MB real sensor data

### Что в данных

**Sensors** (11 каналов):
- **PS1-PS6**: Pressure sensors (main hydraulic circuit)
- **FS1-FS2**: Flow rate
- **EPS1**: Electrical power
- **CE**: Contamination/Erosion
- **CP**: Cooler Power
- **SE**: Specific Energy
- **TS1-TS4**: Temperature
- **VS1**: Vibration

**Duration**: 60 hours of operation (real equipment)

**Sample rate**: Varies by sensor (100 Hz to 10 Hz)

**Labels**: Conditions in profile.txt
- Normal
- Cooler degradation
- Valve leakage
- Pump wear
- Contamination

---

## Проблема #1: Реальные данные не полные

### Что есть

✅ 11 временных рядов от реального оборудования  
✅ 60 часов работы  
✅ Множество нормальных и аномальных состояний  

### Что не хватает

❌ **Разнообразие неисправностей**:
- Данные содержат только 5 типов неисправностей
- Нужны 9 типов (по модели v2.1.0)
- Нет данных о: cavitation, seal wear, bearing fault, valve stuck

❌ **Недостаточно длинные циклы**:
- Нужны полные циклы развития неисправности
- От early degradation до failure
- Текущие данные - только snapshots

❌ **Нет данных по компонентам**:
- Топология гидросистемы известна (из конфигов)
- Но нет per-component sensor data
- Нужны данные для каждого элемента

❌ **Дисбаланс классов**:
- Большинство данных - нормальное состояние
- Редкие аномалии недопредставлены

---

## Решение: Трёхуровневая архитектура данных

### Level 1: Real Data (Существующее)

**Назначение**: Ground truth, калибровка, валидация

```
services/gnn_service/data/
├── raw_real_dataset/          # ✅ Существует (450 MB)
│   └── [UC Irvine data]
├── processed_real/            # ← Создать
│   ├── train_real_60h.pkl
│   ├── val_real_10h.pkl
│   └── test_real_15h.pkl
```

**Pipeline**:
```
Raw sensors → Normalize → Resample → Align time → Split
  (450 MB)     (scaling)   (to 100 Hz) (sync)    (T/V/T)
```

**Output**: 3 datasets (60h train, 10h val, 15h test)

---

### Level 2: Physics-Based Synthetic Data

**Назначение**: Расширить разнообразие неисправностей

#### Архитектура симулятора

```python
# services/gnn_service/src/data_generation/
├── __init__.py
├── hydraulic_simulator.py      # Основной симулятор
├── component_models/
│   ├── pump_model.py          # PDΔ (differential equations)
│   ├── valve_model.py         # Flow control
│   ├── actuator_model.py      # Linear/rotational motion
│   ├── accumulator_model.py   # Energy storage
│   ├── filter_model.py        # Contamination dynamics
│   ├── cooler_model.py        # Heat transfer
│   └── tubing_model.py        # Fluid dynamics
├── fault_models/
│   ├── cavitation.py          # Low pressure, noise
│   ├── leakage.py             # Internal/external losses
│   ├── seal_wear.py           # Increased friction
│   ├── bearing_fault.py       # Vibration, heat
│   ├── valve_stuck.py         # Flow restriction
│   ├── pump_wear.py           # Reduced displacement
│   ├── overheating.py         # Thermal runaway
│   ├── contamination.py       # Particle growth
│   └── pressure_drop.py       # Blockage
└── scenario_generator.py      # Compose scenarios
```

#### Физические модели (ДУ)

**Уравнение потока через помпу**:
```
dP/dt = (K_pump * ω - Q_load) / C_f
```
где:
- K_pump = K_nominal * (1 - degradation_factor)
- Q_load = f(pressure, temperature, contamination)
- C_f = fluid compressibility

**Модель кавитации**:
```
Cavitation = (P_vapor - P_inlet) / P_nominal
Noise_level = σ(Cavitation) * amplitude
```

**Модель износа уплотнения**:
```
Leakage(t) = Leakage_0 + k_wear * wear_rate * t
wear_rate = f(pressure_cycles, seal_material, temperature)
```

**Модель загрязнения**:
```
Contaminant_count(t) = C_0 + ingestion_rate - filtration_rate
Filter_pressure_drop(t) = f(contamination_level, filter_condition)
```

#### Сценарии неисправностей

```python
class FaultScenario:
    def __init__(
        self,
        fault_type: str,           # cavitation, leakage, etc.
        start_time: float,         # When fault begins
        progression_rate: float,   # Degradation speed (hours)
        severity: float,           # Final severity [0, 1]
        environmental_stress: dict # Temperature, contamination
    ):
        pass

# Пример:
scenario = FaultScenario(
    fault_type='seal_wear',
    start_time=10.0,           # Начать на 10-й час
    progression_rate=0.02,     # Медленный износ
    severity=0.85,             # До 85% потери герметичности
    environmental_stress={
        'temperature': 65,      # °C
        'contamination': 5000,  # particles/100mL
        'pressure_cycles': 100, # cycles per minute
    }
)

# Симуляция:
sensor_data = simulator.run(
    scenario=scenario,
    duration_hours=24,
    sensor_noise=0.01,  # 1% white noise
    sample_rate=100,    # Hz
)
```

#### Библиотека сценариев

```
services/gnn_service/data/
└── synthetic_scenarios.json
    ├── cavitation
    │   ├── slow_progression
    │   ├── rapid_onset
    │   └── cyclic_pattern
    ├── leakage
    │   ├── internal_leak
    │   ├── external_leak
    │   └── progressive_leak
    ├── seal_wear
    ├── bearing_fault
    ├── valve_stuck
    ├── overheating
    ├── contamination
    └── pressure_drop (filter clogging)
```

**Output**: 
```
services/gnn_service/data/
└── synthetic_data/
    ├── train_synthetic_500h.pkl  # 500 часов синтетики
    ├── scenarios_metadata.json   # Параметры каждого сценария
    └── fault_library.json        # Библиотека неисправностей
```

---

### Level 3: Semisynthetic Features (КЛЮЧЕВАЯ АРХИТЕКТУРА)

**Концепция**: Недостающие данные = берутся из топологии

#### Проблема

Модель v2.1.0 ожидает:
- **Component-level features**: данные от каждого элемента
- **Edge features**: взаимодействия между компонентами
- **Global features**: общее состояние системы

У нас есть только:
- 11 sensor channels (не покрывают все компоненты)

#### Решение: Feature Imputation из Топологии

```python
# services/gnn_service/src/data/feature_engineering.py

class SemisyntheticFeatureEngine:
    """Генерирует недостающие признаки из топологии системы."""
    
    def __init__(self, topology_config: dict):
        """
        Args:
            topology_config: Из configs/topology_templates.json
                {
                    'components': [
                        {'id': 'pump_01', 'type': 'pump', 'sensors': ['PS1', 'FS1']},
                        {'id': 'valve_01', 'type': 'directional_valve', 'sensors': ['PS2']},
                        ...
                    ],
                    'edges': [
                        {'from': 'pump_01', 'to': 'valve_01', 'type': 'pressure_line'},
                        ...
                    ]
                }
        """
        self.topology = topology_config
        self.component_models = self._init_models()
    
    def fill_missing_sensors(self, observed_data: dict) -> dict:
        """
        Для компонентов без direct sensors, вычислить из соседних.
        
        Логика:
        1. Если у компонента есть sensors → используем их
        2. Если нет → вычисляем из:
           - Физических уравнений (ДУ компонента)
           - Данных соседних компонентов
           - Топологических ограничений
        
        Пример:
            # Если нет данных от cooler_outlet_pressure
            # но есть cooler_inlet_pressure + flow_rate
            outlet_pressure = inlet_pressure - delta_p(flow_rate)
        """
        imputed_data = observed_data.copy()
        
        for component in self.topology['components']:
            component_id = component['id']
            direct_sensors = component.get('sensors', [])
            
            # Какие признаки нужны этому компоненту
            required_features = self._get_required_features(component['type'])
            
            # Какие есть
            available_features = {s: observed_data[s] for s in direct_sensors if s in observed_data}
            
            # Какие не хватает
            missing_features = set(required_features) - set(available_features.keys())
            
            # Вычислить недостающие
            for feature in missing_features:
                value = self._impute_feature(
                    component=component,
                    feature=feature,
                    observed=observed_data,
                    available=available_features
                )
                imputed_data[f"{component_id}_{feature}"] = value
        
        return imputed_data
    
    def _impute_feature(self, component, feature, observed, available):
        """
        Вычислить отсутствующий признак.
        
        Примеры:
        1. Outlet pressure = Inlet pressure - pressure drop
           Δp = f(flow_rate, temperature, component_type)
        
        2. Outlet temperature = Inlet temp + heat_change
           Q = m * c_p * ΔT
        
        3. Outlet flow = Inlet flow - leakage
           Leakage = f(pressure_diff, material, time_degradation)
        """
        
        if component['type'] == 'pump':
            if feature == 'outlet_flow':
                # Q_out = Q_nominal * (1 - degradation)
                degradation = available.get('pressure_ripple', 0) / 100
                return available['inlet_flow'] * (1 - degradation)
            
            elif feature == 'outlet_pressure':
                # P_out = P_in + generated_pressure
                # P_generated зависит от: скорость, износ
                degradation = self._estimate_pump_degradation(component)
                return available['inlet_pressure'] + (100 - degradation)
        
        elif component['type'] == 'directional_valve':
            if feature == 'outlet_pressure':
                # P_out ≈ P_in - ΔP_valve
                pressure_drop = self._calc_valve_pressure_drop(
                    flow=available.get('flow_rate', 0),
                    valve_position=available.get('spool_position', 0.5),
                    contamination=observed.get('CE', 0)  # Загрязнение влияет на потери
                )
                return available['inlet_pressure'] - pressure_drop
        
        elif component['type'] == 'cooler':
            if feature == 'outlet_temperature':
                # T_out = T_in - Q_cooled / (m_dot * c_p)
                heat_removed = available.get('cooler_power', 0)
                return available['inlet_temperature'] - (heat_removed / (available.get('mass_flow') * 1800))
        
        elif component['type'] == 'filter':
            if feature == 'pressure_drop':
                # ΔP_filter = f(contamination_level, filter_condition, flow)
                contamination = observed.get('CE', 0)
                flow = available.get('flow_rate', 0)
                # Степенной закон: ΔP ~ flow^2 * contamination^1.5
                return 0.5 * (flow**2) * ((contamination/1000)**1.5)
        
        return 0.0  # Default
    
    def fill_edge_features(self, node_features: dict) -> dict:
        """
        Для каждого ребра топологии, вычислить признаки.
        
        Edge features:
        - pressure_delta: Разница давлений между компонентами
        - flow_rate: Поток по трубопроводу
        - temperature_delta: Разница температур
        - fluid_power: P * Q (гидравлическая мощность)
        - resistance: ΔP / Q (гидравлическое сопротивление)
        """
        edge_features = {}
        
        for edge in self.topology['edges']:
            edge_id = f"{edge['from']} → {edge['to']}"
            
            from_pressure = node_features.get(f"{edge['from']}_outlet_pressure", 0)
            to_pressure = node_features.get(f"{edge['to']}_inlet_pressure", 0)
            flow = node_features.get(f"{edge['from']}_outlet_flow", 0)
            
            edge_features[edge_id] = {
                'pressure_delta': from_pressure - to_pressure,
                'flow_rate': flow,
                'temperature_delta': node_features.get(f"{edge['from']}_outlet_temp", 0) - 
                                    node_features.get(f"{edge['to']}_inlet_temp", 0),
                'power': abs((from_pressure - to_pressure) * flow),
                'resistance': (from_pressure - to_pressure) / (flow + 1e-6),  # Avoid division by zero
                'flow_direction': edge['type'],
            }
        
        return edge_features
    
    def system_aggregation(self, node_features: dict, edge_features: dict) -> dict:
        """
        Вычислить global features (уровень системы).
        
        Логика: Sum of subsystem powers = Total system power
        (как матрица дифуравнений на уровне всей системы)
        
        Примеры:
        - Total hydraulic power = Σ(P_i * Q_i)
        - Total heat generation = Σ(losses_i)
        - System efficiency = Output power / Input power
        - Overall degradation = mean(component_degradations)
        """
        
        total_power = sum(
            edge_features[edge]['power'] 
            for edge in edge_features
        )
        
        total_heat = sum(
            abs(edge_features[edge]['pressure_delta'] * edge_features[edge]['flow_rate'] * 0.1)
            for edge in edge_features
        )
        
        return {
            'total_power_w': total_power,
            'total_heat_w': total_heat,
            'system_efficiency': (total_power - total_heat) / (total_power + 1e-6),
            'mean_pressure': np.mean([v['outlet_pressure'] for k, v in node_features.items() if 'outlet_pressure' in str(k)]),
            'mean_temperature': np.mean([v for k, v in node_features.items() if 'temperature' in str(k)]),
        }
```

#### Правило целостности

```python
class FeatureIntegrity:
    """
    Проверить, что imputed features физически согласованы.
    """
    
    @staticmethod
    def conservation_of_flow():
        """
        Conservation of mass:
        Σ Q_in = Σ Q_out (для каждого узла)
        """
        for node in topology['components']:
            inflows = []
            outflows = []
            # Проверить
            assert abs(sum(inflows) - sum(outflows)) < tolerance
    
    @staticmethod
    def energy_balance():
        """
        Energy balance:
        Power_in = Power_useful + Power_lost
        """
        # P_in = Σ(P_pump * Q_pump)
        # P_useful = Σ(P_actuator * Q_actuator)
        # P_lost = Σ(ΔP * Q) по всем сопротивлениям
        assert P_in == P_useful + P_lost
    
    @staticmethod
    def pressure_continuity():
        """
        No pressure jumps between connected components.
        P_outlet_A ≈ P_inlet_B (for connected A→B)
        """
        for edge in topology['edges']:
            from_comp, to_comp = edge['from'], edge['to']
            assert abs(
                features[f"{from_comp}_outlet_pressure"] - 
                features[f"{to_comp}_inlet_pressure"]
            ) < tolerance
```

---

## Phase 2-4 Timeline

### Phase 2: Integration Fixes (Jan 3-4) ✅
- 3 code fixes (tuple → dict)
- Runtime: 4-6 hours
- Goal: Model works with inference

### Phase 3A: Real Data Pipeline (Jan 5-12) ⏳
- Load UC Irvine dataset
- Normalize, resample, align
- Train/val/test split
- Create baseline
- **Duration**: 1-2 weeks
- **Output**: Processed real data (train/val/test)

### Phase 3B: Physics-Based Synthetic (Jan 12-26) ⏳
- Implement hydraulic simulator
- Component models (PDEs)
- Fault models (9 types)
- Scenario generator
- Synthetic data generation (500+ hours)
- **Duration**: 2-3 weeks
- **Output**: Synthetic dataset + scenarios

### Phase 3C: Semisynthetic Features (Jan 26-Feb 2) ⏳
- Topology integration
- Feature imputation engine
- Edge feature extraction
- System aggregation
- Integrity validation
- **Duration**: 1 week
- **Output**: Complete feature matrix

### Phase 3D: Full Training Pipeline (Feb 2-9) ⏳
- DataLoader integration
- Batch sampling strategy
- Mixed real+synthetic training
- Validation on real data
- Hyperparameter tuning
- **Duration**: 1 week
- **Output**: Production-ready model

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA ARCHITECTURE                         │
└─────────────────────────────────────────────────────────────┘

    UC IRVINE DATASET
         (450 MB)
           │
           ▼
    ┌──────────────────┐
    │  Phase 3A        │
    │ Real Pipeline    │  ← Normalize, resample, align
    │  Processor       │
    └────────┬─────────┘
             │
      ┌──────┴──────┐
      │             │
      ▼             ▼
   60h train    10h val
   
        ┌──────────────┐
        │  Phase 3B    │
        │   Physics    │  ← Hydraulic equations
        │ Simulator    │  ← 9 fault models
        │              │  ← Scenario library
        └────┬─────────┘
             │
             ▼
      500h synthetic
      (balanced faults)
      
    ┌─────────────────┐
    │  Phase 3C       │
    │ Semisynthetic   │  ← Topology imputation
    │ Feature Engine  │  ← Edge extraction
    │                 │  ← System aggregation
    └────┬────────────┘
         │
    ┌────┴─────────────────────────┐
    │                              │
    ▼                              ▼
Complete node features      Complete edge features
(per component)            (interactions)
    │                              │
    └──────────────┬───────────────┘
                   │
                   ▼
        ┌──────────────────┐
        │  Phase 3D        │
        │ Training         │  ← Mixed real + synthetic
        │ Pipeline         │  ← Stratified sampling
        │                  │  ← Validation on real data
        └──────────────────┘
                   │
                   ▼
          Production Model
        (v2.1.0 trained)
```

---

## Что реализовано vs План

| Компонент | Статус | Примечание |
|-----------|--------|------------|
| **Real data** | ✅ Есть | UC Irvine, 450 MB |
| **Real pipeline** | ❌ TODO | Phase 3A |
| **Simulator** | ❌ TODO | Phase 3B |
| **Fault models** | ❌ TODO | Phase 3B |
| **Semisynthetic** | ❌ TODO | Phase 3C (ключевое!) |
| **DataLoader** | ❌ TODO | Phase 3D |
| **Training** | ❌ TODO | Phase 3D |

---

## Success Criteria

✅ **Real data pipeline**:
- Loads 450 MB dataset
- Preprocesses correctly
- Splits into train/val/test
- No data leakage
- Balanced classes

✅ **Physics simulator**:
- Implements 9 PDEs (pump, valve, cooler, filter, actuator, accumulator, tubing, seal, bearing)
- Generates realistic sensor data
- Matches real data statistics
- Configurable fault scenarios

✅ **Semisynthetic features**:
- Imputes missing sensor data from topology
- Maintains physical consistency
- Edge features computed correctly
- System-level aggregation works
- Validation passes (flow conservation, energy balance, pressure continuity)

✅ **Training**:
- Model trains on mixed real+synthetic
- Validates on held-out real data
- Metrics improve over epochs
- Convergence achieved

---

## References

**UC Irvine Dataset**:
- [Hydraulic Systems Data](https://archive.ics.uci.edu/ml/datasets/Hydraulic+Systems+Predictive+Maintenance+and+Condition+Monitoring)
- 11 sensors, 60 hours, 8 health conditions

**Physics Models**:
- Merritt, H.E. (1967). Hydraulic Control Systems. Wiley.
- ISO 4413: Safety of hydraulic fluid power systems and components
- Rydberg, K.E. (2005). Energy Efficient Hydraulic Actuation Systems

**GNN for Diagnostics**:
- Kipf & Welling (2017): Semi-Supervised Classification with GCNs
- Veličković et al. (2018): Graph Attention Networks
- Hochreiter & Schmidhuber (1997): LSTM networks

---

**Status**: Ready for Phase 3A implementation  
**Next Step**: Start with real data pipeline (Phase 3A)  
**Time**: Start immediately after Phase 2 verification
