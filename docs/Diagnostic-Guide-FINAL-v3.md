# 📚 Diagnostic System Guide v3.0 FINAL

## С Mermaid диаграммами, точками датчиков и реальными схемами

---

## 📖 Содержание

1. [Архитектура системы диагностики](#архитектура-системы)
2. [Точки подключения датчиков](#точки-подключения-датчиков)
3. [Красивые Mermaid диаграммы](#красивые-mermaid-диаграммы)
4. [Три метода диагностики](#три-метода-диагностики)
5. [Django Models](#django-models)
6. [API & Frontend](#api--frontend)
7. [Roadmap](#roadmap)

---

## 🏗️ Архитектура системы

### Общая архитектура

```mermaid
graph TB
    subgraph Frontend["🎨 Frontend (Nuxt 3)"]
        F1["Dashboard"]
        F2["Diagnostics UI"]
        F3["Real-time Charts"]
    end

    subgraph Backend["⚙️ Backend (Django 5.2)"]
        B1["REST API"]
        B2["WebSocket"]
        B3["RAG AI"]
    end

    subgraph Database["💾 Database (TimescaleDB)"]
        D1["Sensor Data"]
        D2["Analysis Results"]
        D3["System Config"]
    end

    subgraph ML["🤖 AI/ML (Ollama + Qwen3)"]
        M1["LLM Engine"]
        M2["FAISS Index"]
        M3["RAG Pipeline"]
    end

    F1 -->|HTTP/WS| B1
    F2 -->|HTTP/WS| B2
    F3 -->|Real-time| B2

    B1 --> D1
    B1 --> D2
    B3 --> M1
    M1 --> M2
    M2 --> M3
    B3 --> M3

    style Frontend fill:#e1f5fe
    style Backend fill:#f3e5f5
    style Database fill:#e8f5e8
    style ML fill:#fff3e0
```

---

## 📍 Точки подключения датчиков

### Стенд Дубовика (с рекуперацией энергии)

```
┌─────────────────────────────────────────────────────────────┐
│                    ИСПЫТАТЕЛЬНЫЙ СТЕНД                       │
└─────────────────────────────────────────────────────────────┘

    [ЭЛЕКТРОДВИГАТЕЛЬ]
            ↓ ω_н (об/мин)
         📊 ДАТЧИК 1
            ↓
    [МУФТА МНОГОДИСКОВАЯ]
            ↓
    [ГИДРОНАСОС] ← ДАТЧИК P₁ (Мпа) [Давление сети]
         ↓
         ├─ ДАТЧИК Q₁ (л/мин) [Расход]
         └─ ДАТЧИК T₁ (°C) [Температура]
         ↓
    [ОБРАТНЫЙ КЛАПАН]
         ↓
    ┌────────────────────────────────────────┐
    │      ТОЧКА 2 (Давление после клапана)  │
    │  📊 ДАТЧИК P₂ (Мпа)                   │
    └────────────────────────────────────────┘
         ↓
         ├──────────────────┬──────────────────┐
         ↓                  ↓                  ↓
    ┌─────────┐       ┌─────────┐       ┌─────────┐
    │ ТОЧКА 3 │       │ ТОЧКА 4 │       │ ТОЧКА 7 │
    │ (P₃)    │       │ (P₄)    │       │ (Слив)  │
    │ 📊 ПОК  │       │ 📊 ПОК  │       │ 📊 ПОК  │
    └─────────┘       └─────────┘       └─────────┘
         ↓                  ↓                  ↓
    [ПРЕДОХР.         [ГИДРОМОТОР]       [БАК]
     КЛАПАН]               ↓
                       📊 ДАТЧИК 2
                       ω_м (об/мин)

    [ФИЛЬТР СЛИВА]
         ↓
    [ТЕПЛООБМЕННИК]
         ↓
    [АККУМУЛЯТОР (опционально)]
         ↓
    [ГИДРОБАК]
```

### Матрица датчиков (реальная конфигурация)

| Точка | Датчик ID     | Тип         | Единица | Диапазон     | Назначение                   |
| ----- | ------------- | ----------- | ------- | ------------ | ---------------------------- |
| **1** | P_pump        | Давление    | Мпа     | 0-250        | Давление на выходе насоса    |
| **1** | Q_pump        | Расход      | л/мин   | 0-50         | Расход насоса                |
| **1** | T_pump        | Температура | °C      | -10 до +80   | Температура жидкости         |
| **2** | P_after_valve | Давление    | Мпа     | 0-250        | После обратного клапана      |
| **3** | P_relief      | Давление    | Мпа     | 0-250        | На предохранительном клапане |
| **4** | P_motor_inlet | Давление    | Мпа     | 0-250        | На входе мотора              |
| **M** | ω_motor       | Скорость    | об/мин  | 0-3000       | Угловая скорость мотора      |
| **M** | F_motor       | Усилие      | Н       | 0-10000      | Усилие на валу мотора        |
| **5** | P_return      | Давление    | Мпа     | 0-10         | Давление в сливе             |
| **6** | x_position    | Положение   | мм      | 0-100        | Положение штока цилиндра     |
| **6** | V_piston      | Скорость    | мм/с    | -500 до +500 | Скорость поршня              |
| **7** | Vibr_accel    | Вибрация    | g       | 0-10         | Ускорение (вибрация)         |

---

## 🎨 Красивые Mermaid диаграммы

### 1. Диагностический цикл (главный процесс)

```mermaid
graph TD
    A["🔴 НАЧАЛО ДИАГНОСТИКИ"] --> B["📊 СБОР ДАННЫХ<br/>• Датчики<br/>• История<br/>• Пробы"]

    B --> C1["1️⃣ МАТЕМАТ. МОДЕЛЬ<br/>вес: 40%"]
    B --> C2["2️⃣ ФАЗОВЫЕ ПОРТРЕТЫ<br/>вес: 40%"]
    B --> C3["3️⃣ ТРИБОДИАГНОСТИКА<br/>вес: 20%"]

    C1 --> D["🔄 НОРМАЛИЗАЦИЯ<br/>Оценки 0.0-1.0"]
    C2 --> D
    C3 --> D

    D --> E["🎯 ИНТЕГРАЛЬНАЯ ОЦЕНКА<br/>D = 0.4×M + 0.4×P + 0.2×T"]

    E --> F{D - score?}

    F -->|D < 0.3| G["✅ НОРМА"]
    F -->|0.3 ≤ D < 0.6| H["⚠️ ПРЕДУПРЕЖДЕНИЕ"]
    F -->|D ≥ 0.6| I["🔴 НЕИСПРАВНОСТЬ"]

    G --> J["📋 ФОРМИРОВАНИЕ ОТЧЁТА"]
    H --> J
    I --> J

    J --> K["💾 СОХРАНЕНИЕ<br/>в БД"]

    K --> L["🟢 КОНЕЦ"]

    style A fill:#ff6b6b
    style G fill:#51cf66
    style H fill:#ffd43b
    style I fill:#ff6b6b
    style L fill:#4c6ef5
```

### 2. Математическая модель (алгоритм)

```mermaid
flowchart TD
    A["START<br/>Получить параметры"] --> B["📈 ИЗМЕРЕННЫЕ ДАННЫЕ<br/>P₁-P₄, Q₁-Q₂₋₄<br/>ω_н, ω_м, T"]

    B --> C["🧮 РАСЧЁТ ПО МОДЕЛИ<br/>M_н = V × P / 2π×η<br/>ω_ожид = ΔM / I<br/>Q_ожид = ω × V × η"]

    C --> D["⚖️ СРАВНЕНИЕ<br/>ΔP = |P_изм - P_расч| / P_расч<br/>ΔQ = |Q_изм - Q_расч| / Q_расч"]

    D --> E["📊 Δ_общ = MAX(ΔP, ΔQ)"]

    E --> F{Δ%?}

    F -->|< 5%| G["✅ НОРМА"]
    F -->|5-10%| H["⚠️ ПРЕДУПРЕЖДЕНИЕ"]
    F -->|≥ 10%| I["🔴 НЕИСПРАВНОСТЬ"]

    G --> J["score = 0.1"]
    H --> J["score = 0.5"]
    I --> J["score = 0.9"]

    J --> K["END<br/>Return score"]

    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style G fill:#c8e6c9
    style H fill:#fff9c4
    style I fill:#ffcccc
```

### 3. Фазовые портреты (анализ)

```mermaid
flowchart TD
    A["START<br/>Регистрация данных 20+ сек"] --> B["📊 Собрать временные ряды<br/>x(t), V(t), F(t), P(t)<br/>Частота: 100 Гц"]

    B --> C["🎨 ПОСТРОЕНИЕ ФП<br/>V=fx, F=fV, P=fQ"]

    C --> D["📐 РАСЧЁТ ПЛОЩАДИ<br/>S = 0.5×|Σ(xᵢ×yᵢ₊₁) - ...|<br/>по формуле Гаусса"]

    D --> E["📊 ΔS = |S - S_этал| / S_этал"]

    E --> F{ΔS%?}

    F -->|< 10%| G["✅ НОРМА<br/>score=0.1"]
    F -->|10-25%| H["⚠️ ПРЕДОТКАЗ<br/>score=0.5"]
    F -->|≥ 25%| I["🔴 ОТКАЗ<br/>score=0.9"]

    G --> J["🔍 АНАЛИЗ ФОРМЫ<br/>Смещение центра?<br/>Разрывы контура?<br/>Искажение?"]
    H --> J
    I --> J

    J --> K["END<br/>Return score + рекомендации"]

    style A fill:#e3f2fd
    style C fill:#f3e5f5
    style D fill:#ede7f6
```

### 4. Трибодиагностика (анализ жидкости)

```mermaid
flowchart TD
    A["START<br/>Отбор пробы 100 мл"] --> B["🧪 ЛАБОРАТОРНЫЙ АНАЛИЗ"]

    B --> C1["🔬 СЧЁТЧИК ЧАСТИЦ<br/>≥4 мкм, ≥6 мкм, ≥14 мкм"]
    B --> C2["⚗️ СПЕКТРОМЕТР<br/>Fe, Cu, Al, Cr, Si"]
    B --> C3["📏 ВИСКОЗИМЕТР<br/>сСт, pH, вода ppm"]

    C1 --> D["📊 ISO 4406 класс<br/>N₁/N₂/N₃"]
    C2 --> E["🔍 ИСТОЧНИК ИЗНОСА<br/>Fe↑ = насос<br/>Cu↑ = подшипники<br/>Al↑ = цилиндры"]
    C3 --> F["⚠️ СОСТОЯНИЕ ЖИД<br/>pH < 6 = окисление<br/>воде > 500ppm = проблема"]

    D --> G{ISO класс?}
    E --> H{Элемент?}
    F --> I{Состояние?}

    G -->|15/13/10| G1["✅ Норма<br/>score=0.1"]
    G -->|16/14/11| G2["⚠️ Внимание<br/>score=0.5"]
    G -->|≥17/15/12| G3["🔴 Критика<br/>score=0.9"]

    G1 --> J["🎯 ФИНАЛЬНАЯ ОЦЕНКА"]
    G2 --> J
    G3 --> J

    H --> J
    I --> J

    J --> K["END<br/>Return score + рекомендации"]

    style A fill:#e3f2fd
    style B fill:#f3e5f5
    style G1 fill:#c8e6c9
    style G2 fill:#fff9c4
    style G3 fill:#ffcccc
```

### 5. Интегральная оценка (финал)

```mermaid
graph LR
    A["📊 Math Score<br/>M = 0.1-0.9"] --> D["🎯 ИНТЕГРАЛЬНАЯ<br/>D = 0.4×M + 0.4×P + 0.2×T"]
    B["📉 Phase Score<br/>P = 0.1-0.9"] --> D
    C["🧪 Tribo Score<br/>T = 0.1-0.9"] --> D

    D --> E{D?}

    E -->|0.0-0.3| F["✅ НОРМА<br/>Продолжить мониторинг"]
    E -->|0.3-0.6| G["⚠️ ПРЕДУПРЕЖДЕНИЕ<br/>Запланировать обслуживание"]
    E -->|0.6-1.0| H["🔴 НЕИСПРАВНОСТЬ<br/>Немедленный ремонт"]

    F --> I["📋 ОТЧЁТ<br/>• Статус<br/>• Детали<br/>• Рекомендации<br/>• Ресурс"]
    G --> I
    H --> I

    style A fill:#e1f5fe
    style B fill:#e1f5fe
    style C fill:#e1f5fe
    style D fill:#fff3e0
    style F fill:#c8e6c9
    style G fill:#fff9c4
    style H fill:#ffcccc
    style I fill:#f3e5f5
```

---

## 🔧 Три метода диагностики

### Метод 1️⃣: Математическое моделирование

**Формулы:**

\(
M_н = \frac{V_0 \times P}{2\pi \times \eta}
\)

\(
\omega = \frac{M*д + M*м \times n - M*н}{I*д + I*н + I*{мех}}
\)

\(
Q = \omega \times V_0 \times \eta
\)

**Диагностика:**

- Δ < 5% → ✅ Норма
- 5% ≤ Δ < 10% → ⚠️ Внимание
- Δ ≥ 10% → 🔴 Неисправность

---

### Метод 2️⃣: Фазовые портреты

**Формула Гаусса (площадь):**

\(
S = 0.5 \times \left| \sum*{i=1}^{n-1} x_i y*{i+1} + x*n y_1 - \sum*{i=1}^{n-1} x\_{i+1} y_i - x_1 y_n \right|
\)

**Анализ:**
| ΔS | Статус | Действие |
|----|--------|----------|
| < 10% | ✅ Норма | Продолжить |
| 10-25% | ⚠️ Предотказ | Обслуживание |
| > 25% | 🔴 Отказ | Ремонт |

---

### Метод 3️⃣: Трибодиагностика

**Параметры:**

- ISO 4406 класс чистоты
- Элементный состав (Fe, Cu, Al, Cr)
- Физико-химические свойства (pH, вязкость, вода)

**Норма ISO 15/13/10**

---

## 💻 Django Models

### Core Models

```python
# backend/apps/diagnostics/models/core.py

class HydraulicSystem(models.Model):
    """Гидравлическая система"""
    name = models.CharField(max_length=200)
    system_type = models.CharField(max_length=20)
    max_pressure = models.FloatField()  # МПа
    nominal_flow = models.FloatField()  # л/мин
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'hydraulic_systems'

class Equipment(models.Model):
    """Оборудование"""
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    equipment_type = models.CharField(max_length=20)  # pump, motor, valve...
    technical_parameters = models.JSONField(default=dict)
    installation_date = models.DateTimeField()

    class Meta:
        db_table = 'equipments'

class Sensor(models.Model):
    """Датчик"""
    equipment = models.ForeignKey(Equipment, on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    sensor_type = models.CharField(max_length=20)  # pressure, flow, temp...
    unit = models.CharField(max_length=20)  # MPa, l/min, °C...
    min_value = models.FloatField()
    max_value = models.FloatField()
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = 'sensors'
```

### Diagnostic Results

```python
class MathematicalModelResult(models.Model):
    """Результат математической модели"""
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()

    status = models.CharField(max_length=20)  # normal, warning, fault
    deviation = models.FloatField()  # Δ%
    score = models.FloatField()  # 0.0-1.0

    class Meta:
        db_table = 'math_model_results'
        ordering = ['-timestamp']

class PhasePortraitResult(models.Model):
    """Результат фазовых портретов"""
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()

    area = models.FloatField()
    area_deviation = models.FloatField()  # ΔS%
    status = models.CharField(max_length=20)
    score = models.FloatField()  # 0.0-1.0

    class Meta:
        db_table = 'phase_portrait_results'

class IntegratedDiagnosticResult(models.Model):
    """Интегральный результат"""
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE)
    timestamp = models.DateTimeField()

    math_score = models.FloatField()
    phase_score = models.FloatField()
    tribo_score = models.FloatField()

    integrated_score = models.FloatField()  # D
    overall_status = models.CharField(max_length=20)

    recommendations = models.TextField()
    predicted_remaining_life = models.IntegerField()  # часов

    class Meta:
        db_table = 'integrated_results'
        ordering = ['-timestamp']
```

---

## 🎨 API & Frontend

### API Endpoints

```
POST   /api/diagnostics/mathematical-model/
       Input: {system_id}
       Output: {status, deviation, score, recommendations}

POST   /api/diagnostics/phase-portraits/
       Input: {system_id}
       Output: {area, deviation, status, score}

POST   /api/diagnostics/tribodiagnostics/
       Input: {sample_id}
       Output: {iso_class, elements, status, score}

POST   /api/diagnostics/integrated/
       Input: {system_id}
       Output: {integrated_score, overall_status, recommendations}
```

### Frontend Pages

```
pages/
├── diagnostics/
│   ├── index.vue                    # Главная
│   ├── mathematical-model.vue       # Мат модель
│   ├── phase-portraits.vue          # ФП
│   ├── tribodiagnostics.vue         # Трибо
│   └── integrated.vue               # Интегральный анализ
├── systems/[systemId]/
│   └── index.vue                    # Обзор системы
└── reports.vue                      # Отчёты
```

---

## 🚀 Roadmap

### Sprint 1: Models (1 неделя)

- [ ] Django models
- [ ] Миграции
- [ ] DRF serializers

### Sprint 2: API (1 неделя)

- [ ] Endpoints всех методов
- [ ] Calculations functions
- [ ] WebSocket для real-time

### Sprint 3: Frontend (2 недели)

- [ ] Pages создание
- [ ] VEchart интеграция
- [ ] Forms

### Sprint 4: Testing (1 неделя)

- [ ] Unit tests
- [ ] Integration tests
- [ ] E2E тесты

---

**Версия:** v3.0 FINAL ✅  
**Статус:** Готово к разработке  
**Сложность:** ⭐⭐⭐⭐ (средняя-высокая)

**Источники:**

- Дубовик Е.А. "Стенд для испытания объемных гидромашин с рекуперацией энергии"
- Гареев А.М. "Использование фазовых портретов для диагностирования гидравлических систем"
- Галдин Н.С., Семенова И.А. "Гидравлические схемы мобильных машин"
- ГОСТ 2.704-76 "Правила выполнения гидравлических схем"
