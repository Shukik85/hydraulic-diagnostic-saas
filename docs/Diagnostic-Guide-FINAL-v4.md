# 📚 Diagnostic System Guide v4.0 FINAL
## Актуализированное руководство по системе диагностики

---

## 📝 Содержание

1. [Архитектура системы диагностики](#архитектура-системы)
2. [Методы диагностики](#методы-диагностики)
3. [Процесс диагностики](#процесс-диагностики)
4. [Django Models](#django-models)
5. [API & Frontend](#api--frontend)
6. [Roadmap](#roadmap)

---

## 🏠 Архитектура системы

### Общая архитектура

Наша система состоит из четырех основных компонентов:

**🎨 Frontend (Nuxt 4)**
- Dashboard - главная панель с обзором систем
- Diagnostics UI - интерфейс диагностики
- Real-time Charts - графики в реальном времени

**⚙️ Backend (Django 5.2)**
- REST API - основное API для работы с данными
- WebSocket - для real-time обновлений
- RAG AI - система интеллектуальных ответов

**💾 Database (TimescaleDB)**
- Sensor Data - данные с датчиков
- Analysis Results - результаты анализа
- System Config - конфигурации систем

**🤖 AI/ML (Ollama + Qwen3)**
- LLM Engine - языковая модель
- FAISS Index - векторные индексы
- RAG Pipeline - конвейер обработки

### Потоки данных

1. **Сбор данных**: Датчики → TimescaleDB
2. **Обработка**: TimescaleDB → Django AI Engine
3. **Анализ**: AI Engine → 3 метода диагностики
4. **Отображение**: Django → WebSocket → Nuxt 4 Frontend
5. **Консультации**: RAG System → Frontend Chat

---

## 🔧 Методы диагностики

Наша система использует три независимых метода диагностики, которые объединяются в интегральную оценку.

### Метод 1️⃣: Математическое моделирование (вес: 40%)

**Описание**: Сравниваем реальные показания датчиков с теоретическими значениями, рассчитанными по математическим моделям гидравлических систем.

**Основные формулы:**

1. **Момент насоса:**
   \(M_н = \frac{V_0 \times P}{2\pi \times \eta}\)
   
2. **Угловая скорость:**
   \(\omega = \frac{M_д + M_м \times n - M_н}{I_д + I_н + I_{\text{мех}}}\)
   
3. **Расход:**
   \(Q = \omega \times V_0 \times \eta\)

**Критерии оценки:**
- Δ < 5% → ✅ **Норма** (score = 0.1)
- 5% ≤ Δ < 10% → ⚠️ **Предупреждение** (score = 0.5)
- Δ ≥ 10% → 🔴 **Неисправность** (score = 0.9)

**Особенности реализации:**
- Используем окно данных 20-30 секунд
- Вычисляем средние значения для стабильности
- Учитываем погрешности датчиков

### Метод 2️⃣: Фазовые портреты (вес: 40%)

**Описание**: Анализируем траектории движения в фазовом пространстве, строя графики зависимости скорости от положения, силы от скорости и давления от расхода.

**Основная формула (площадь по Гауссу):**
\(S = 0.5 \times \left| \sum_{i=1}^{n-1} x_i y_{i+1} + x_n y_1 - \sum_{i=1}^{n-1} x_{i+1} y_i - x_1 y_n \right|\)

**Отклонение площади:**
\(ΔS = \frac{|S - S_{этал}|}{S_{этал}} \times 100\%\)

**Критерии оценки:**
| ΔS | Статус | Действие | Score |
|----|--------|---------|---------|
| < 10% | ✅ Норма | Продолжить | 0.1 |
| 10-25% | ⚠️ Предотказ | Обслуживание | 0.5 |
| > 25% | 🔴 Отказ | Немедленный ремонт | 0.9 |

**Типы фазовых портретов:**
- **V = f(x)**: скорость от положения
- **F = f(V)**: сила от скорости  
- **P = f(Q)**: давление от расхода

**Особенности реализации:**
- Сбор данных в течение 20+ секунд с частотой 100 Гц
- Анализ формы контура (разрывы, смещения, искажения)
- Определение центра масс и эксцентриситета

### Метод 3️⃣: Трибодиагностика (вес: 20%)

**Описание**: Лабораторный анализ проб рабочей жидкости для определения состояния и степени износа компонентов гидросистемы.

**Основные параметры анализа:**

1. **Счетчик частиц (ISO 4406)**:
   - ≥ 4 мкм/мл
   - ≥ 6 мкм/мл  
   - ≥ 14 мкм/мл

2. **Спектральный анализ (элементы износа)**:
   - **Fe** (железо) → износ насоса/мотора
   - **Cu** (медь) → износ подшипников
   - **Al** (алюминий) → износ цилиндров/поршней
   - **Cr** (хром) → износ уплотнений
   - **Si** (кремний) → загрязнение извне

3. **Физико-химические свойства**:
   - Вязкость при 40°C (сСт)
   - pH уровень
   - Содержание воды (ppm)
   - Кислотное число (мг KOH/г)

**Критерии оценки:**

| Параметр | Норма | Внимание | Критика |
|-----------|------|----------|----------|
| **ISO 4406** | 15/13/10 | 16/14/11 | ≥ 17/15/12 |
| **Score** | 0.1 | 0.5 | 0.9 |

**Диагностические признаки:**
- **Fe↑** = повышенный износ насоса/мотора
- **Cu↑** = износ подшипников или бронзовых втулок
- **Al↑** = износ алюминиевых компонентов (цилиндры, поршни)
- **pH < 6** = окисление масла
- **Вода > 500 ppm** = проблемы с герметичностью

---

## 🔄 Процесс диагностики

### Основной цикл диагностики

**Шаг 1: Начало диагностики**
- Инициация процесса (автоматически или вручную)
- Проверка доступности данных датчиков
- Определение временного окна для анализа

**Шаг 2: Сбор данных**
- Данные с датчиков (давление, расход, температура, скорость)
- Исторические данные из TimescaleDB
- Пробы рабочей жидкости (при наличии)

**Шаг 3: Параллельное выполнение трех методов**
- **Метод 1**: Математическая модель (вес 40%)
- **Метод 2**: Фазовые портреты (вес 40%)
- **Метод 3**: Трибодиагностика (вес 20%)

**Шаг 4: Нормализация оценок**
- Приведение всех оценок к диапазону 0.0-1.0
- Проверка корректности расчетов
- Обработка ошибок и пропущенных данных

**Шаг 5: Интегральная оценка**
\(D = 0.4 \times M + 0.4 \times P + 0.2 \times T\)

где:
- M = оценка математической модели
- P = оценка фазовых портретов
- T = оценка трибодиагностики

**Шаг 6: Определение статуса**
- **D < 0.3** → ✅ **Норма** (продолжить мониторинг)
- **0.3 ≤ D < 0.6** → ⚠️ **Предупреждение** (запланировать обслуживание)
- **D ≥ 0.6** → 🔴 **Неисправность** (немедленный ремонт)

**Шаг 7: Формирование отчета**
- Статус системы
- Детализированные результаты каждого метода
- Конкретные рекомендации по обслуживанию/ремонту
- Прогноз остаточного ресурса (в часах работы)

**Шаг 8: Сохранение в базе данных**
- Запись результатов в TimescaleDB
- Обновление статуса системы
- Отправка уведомлений (при необходимости)

### Алгоритм математической модели

1. **Получить параметры системы**
2. **Собрать измеренные данные**: P₁-P₄, Q₁-Q₄, ωₙ, ωₘ, T
3. **Выполнить расчет по моделям**:
   - Mₙ = V × P / 2π×η
   - ωₒₘᵢₑ = ΔM / I
   - Qₒₘᵢₑ = ω × V × η
4. **Сравнить с измеренными**:
   - ΔP = |Pᵢ₂ₘ - Pᵣₐₛᵩ| / Pᵣₐₛᵩ
   - ΔQ = |Qᵢ₂ₘ - Qᵣₐₛᵩ| / Qᵣₐₛᵩ
5. **Определить общее отклонение**: Δ_общ = MAX(ΔP, ΔQ)
6. **Определить score на основе Д%**

### Алгоритм фазовых портретов

1. **Собрать временные ряды** (20+ сек, 100 Гц): x(t), V(t), F(t), P(t)
2. **Построить фазовые портреты**: V=f(x), F=f(V), P=f(Q)
3. **Рассчитать площадь** по формуле Гаусса
4. **Определить отклонение**: ΔS = |S - Sэтал| / Sэтал
5. **Проанализировать форму** контура:
   - Смещение центра?
   - Разрывы контура?
   - Искажения формы?
6. **Определить score** и сформировать рекомендации

### Алгоритм трибодиагностики

1. **Отобрать пробу** (100 мл рабочей жидкости)
2. **Выполнить лабораторный анализ**:
   - Счетчик частиц (≥ 4, 6, 14 мкм)
   - Спектрометр (Fe, Cu, Al, Cr, Si)
   - Вискозиметр (сСт, pH, вода ppm)
3. **Определить ISO 4406 класс** чистоты
4. **Проанализировать источники износа** по элементному составу
5. **Оценить состояние жидкости** по физико-химическим свойствам
6. **Определить score** и сформировать рекомендации

---

## 💻 Django Models

### Основные модели

```python
# backend/apps/diagnostics/models/core.py
from django.db import models
from django.contrib.auth.models import User
from timescale.db.models.models import TimescaleModel
from timescale.db.models.fields import TimescaleDateTimeField
import uuid

class HydraulicSystem(models.Model):
    """Гидравлическая система"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200, verbose_name="Название системы")
    system_type = models.CharField(
        max_length=50,
        choices=[
            ('test_rig', 'Испытательный стенд'),
            ('production', 'Производственная линия'),
            ('mobile', 'Мобильная техника'),
            ('stationary', 'Стационарная установка'),
        ],
        default='production'
    )
    max_pressure = models.FloatField(verbose_name="Максимальное давление, МПа")
    nominal_flow = models.FloatField(verbose_name="Номинальный расход, л/мин")
    fluid_type = models.CharField(max_length=100, verbose_name="Тип рабочей жидкости")
    owner = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Владелец")
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True, verbose_name="Активна")
    
    class Meta:
        db_table = 'hydraulic_systems'
        verbose_name = "Гидравлическая система"
        verbose_name_plural = "Гидравлические системы"
        ordering = ['-created_at']

class Equipment(models.Model):
    """Оборудование"""
    EQUIPMENT_TYPES = [
        ('pump', 'Гидронасос'),
        ('motor', 'Гидромотор'),
        ('cylinder', 'Гидроцилиндр'),
        ('valve_relief', 'Предохранительный клапан'),
        ('valve_check', 'Обратный клапан'),
        ('accumulator', 'Гидроаккумулятор'),
        ('filter', 'Фильтр'),
        ('tank', 'Гидробак'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE, related_name='equipment')
    name = models.CharField(max_length=200, verbose_name="Название оборудования")
    equipment_type = models.CharField(max_length=50, choices=EQUIPMENT_TYPES)
    technical_parameters = models.JSONField(default=dict, verbose_name="Технические параметры")
    installation_date = models.DateTimeField(verbose_name="Дата установки")
    is_active = models.BooleanField(default=True, verbose_name="Активно")
    
    class Meta:
        db_table = 'equipment'
        unique_together = ['system', 'name']

class Sensor(models.Model):
    """Датчик"""
    SENSOR_TYPES = [
        ('pressure', 'Датчик давления'),
        ('flow', 'Датчик расхода'),
        ('temperature', 'Датчик температуры'),
        ('speed', 'Датчик скорости'),
        ('position', 'Датчик положения'),
        ('vibration', 'Датчик вибрации'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    equipment = models.ForeignKey(Equipment, on_delete=models.CASCADE, related_name='sensors')
    name = models.CharField(max_length=200, verbose_name="Название датчика")
    sensor_type = models.CharField(max_length=50, choices=SENSOR_TYPES)
    sensor_id_external = models.CharField(max_length=100, verbose_name="Внешний ID")
    unit = models.CharField(max_length=20, verbose_name="Единица измерения")
    min_value = models.FloatField(verbose_name="Минимальное значение")
    max_value = models.FloatField(verbose_name="Максимальное значение")
    accuracy = models.FloatField(verbose_name="Точность, %")
    is_active = models.BooleanField(default=True, verbose_name="Активен")
    
    class Meta:
        db_table = 'sensors'
        unique_together = ['equipment', 'sensor_id_external']
```

### Модели диагностики

```python
# backend/apps/diagnostics/models/results.py
class MathematicalModelResult(TimescaleModel):
    """Результат математической модели"""
    time = TimescaleDateTimeField(interval="1 day")
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE)
    
    # Измеренные параметры
    measured_pressure = models.JSONField(verbose_name="Измеренные давления")
    measured_flow = models.JSONField(verbose_name="Измеренные расходы")
    measured_speed = models.JSONField(verbose_name="Измеренные скорости")
    
    # Расчетные параметры
    calculated_pressure = models.JSONField(verbose_name="Расчетные давления")
    calculated_flow = models.JSONField(verbose_name="Расчетные расходы")
    calculated_speed = models.JSONField(verbose_name="Расчетные скорости")
    
    # Отклонения
    overall_deviation = models.FloatField(verbose_name="Общее отклонение, %")
    status = models.CharField(
        max_length=20,
        choices=[
            ('normal', 'Норма'),
            ('warning', 'Предупреждение'),
            ('fault', 'Неисправность'),
        ]
    )
    score = models.FloatField(verbose_name="Оценка (0.0-1.0)")
    
    class Meta:
        db_table = 'mathematical_model_results'
        ordering = ['-time']

class PhasePortraitResult(TimescaleModel):
    """Результат фазовых портретов"""
    time = TimescaleDateTimeField(interval="1 day")
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE)
    
    portrait_type = models.CharField(
        max_length=50,
        choices=[
            ('velocity_position', 'V=f(x)'),
            ('force_velocity', 'F=f(V)'),
            ('pressure_flow', 'P=f(Q)'),
        ]
    )
    
    area = models.FloatField(verbose_name="Площадь портрета")
    reference_area = models.FloatField(verbose_name="Эталонная площадь")
    area_deviation = models.FloatField(verbose_name="Отклонение площади, %")
    
    status = models.CharField(
        max_length=20,
        choices=[
            ('normal', 'Норма'),
            ('pre_fault', 'Предотказ'),
            ('fault', 'Отказ'),
        ]
    )
    score = models.FloatField(verbose_name="Оценка (0.0-1.0)")
    
    portrait_data = models.JSONField(verbose_name="Данные портрета")
    
    class Meta:
        db_table = 'phase_portrait_results'
        ordering = ['-time']

class IntegratedDiagnosticResult(TimescaleModel):
    """Интегральный результат"""
    time = TimescaleDateTimeField(interval="1 day")
    system = models.ForeignKey(HydraulicSystem, on_delete=models.CASCADE)
    
    # Компоненты интегральной оценки
    mathematical_score = models.FloatField(verbose_name="M - математическая модель")
    phase_portrait_score = models.FloatField(verbose_name="P - фазовые портреты")
    tribodiagnostic_score = models.FloatField(verbose_name="T - трибодиагностика")
    
    # Интегральная оценка
    integrated_score = models.FloatField(verbose_name="D - интегральная оценка")
    overall_status = models.CharField(
        max_length=20,
        choices=[
            ('normal', 'Норма'),
            ('warning', 'Предупреждение'),
            ('fault', 'Неисправность'),
        ]
    )
    
    predicted_remaining_life_hours = models.IntegerField(
        null=True, blank=True,
        verbose_name="Остаточный ресурс, часов"
    )
    recommendations = models.TextField(verbose_name="Рекомендации")
    
    class Meta:
        db_table = 'integrated_diagnostic_results'
        ordering = ['-time']
    
    def save(self, *args, **kwargs):
        """Автоматический расчет D = 0.4×M + 0.4×P + 0.2×T"""
        if all([self.mathematical_score is not None, 
               self.phase_portrait_score is not None,
               self.tribodiagnostic_score is not None]):
            self.integrated_score = (
                0.4 * self.mathematical_score +
                0.4 * self.phase_portrait_score +
                0.2 * self.tribodiagnostic_score
            )
            
            # Определение статуса
            if self.integrated_score < 0.3:
                self.overall_status = 'normal'
            elif self.integrated_score < 0.6:
                self.overall_status = 'warning'
            else:
                self.overall_status = 'fault'
        
        super().save(*args, **kwargs)
```

---

## 🖥️ API & Frontend

### API Endpoints

**Основные эндпоинты:**
```
POST   /api/diagnostics/mathematical-model/
       Input: {system_id, time_window}
       Output: {status, deviation, score, recommendations}

POST   /api/diagnostics/phase-portraits/
       Input: {system_id, portrait_type, time_window}
       Output: {area, deviation, status, score, portrait_data}

POST   /api/diagnostics/tribodiagnostics/
       Input: {sample_id, analysis_data}
       Output: {iso_class, elements, status, score, recommendations}

POST   /api/diagnostics/integrated/
       Input: {system_id}
       Output: {integrated_score, overall_status, recommendations, predicted_life}

GET    /api/systems/{id}/diagnostic-history/
       Output: история диагностики за период
```

### Nuxt 4 Frontend страницы

**Структура pages/:**
```
pages/
├── diagnostics/
│   ├── index.vue                    # Главная диагностики
│   ├── mathematical-model.vue       # Математическая модель
│   ├── phase-portraits.vue          # Фазовые портреты
│   ├── tribodiagnostics.vue         # Трибодиагностика
│   └── integrated.vue               # Интегральный анализ
├── systems/[systemId]/
│   └── index.vue                    # Обзор системы
└── reports/
    └── index.vue                    # Отчеты
```

**Компоненты:**
- **DiagnosticChart.vue** - графики результатов
- **PhasePortraitCanvas.vue** - отображение фазовых портретов
- **TriboAnalysisTable.vue** - таблица результатов анализа
- **IntegratedScoreGauge.vue** - индикатор интегральной оценки

---

## 🚀 Roadmap

### Sprint 1: Основа (1 неделя)
- [x] Django 5.2 модели
- [x] TimescaleDB миграции
- [x] DRF сериализаторы
- [ ] Базовые API endpoints

### Sprint 2: Алгоритмы (1 неделя)
- [ ] Математическая модель
- [ ] Алгоритм фазовых портретов
- [ ] Трибодиагностика
- [ ] Интегральная оценка

### Sprint 3: Frontend (2 недели)
- [ ] Nuxt 4 страницы
- [ ] Chart.js интеграция
- [ ] WebSocket работа в реальном времени
- [ ] Мобильная версия

### Sprint 4: Тестирование и деплой (1 неделя)
- [ ] Unit тесты для всех алгоритмов
- [ ] Integration тесты API
- [ ] E2E тесты Nuxt 4 приложения
- [ ] Docker контейнеризация
- [ ] CI/CD пайплайн

---

**Версия:** v4.0 FINAL ✅  
**Статус:** Обновлено и готово к разработке  
**Сложность:** ⭐⭐⭐⭐ (средняя-высокая)
**Обновления v4.0:**
- ✅ Удалены Mermaid диаграммы
- ✅ Удален раздел о точках подключения датчиков
- ✅ Обновлено на Nuxt 4
- ✅ Улучшена читаемость и структурированность