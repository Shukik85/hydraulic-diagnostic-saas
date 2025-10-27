# 📚 Diagnostic System Guide v4.2 FINAL
## Актуализированное руководство по системе диагностики (восстановлена полная структура + корректные формулы)

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
- Dashboard — главная панель с обзором систем
- Diagnostics UI — интерфейс диагностики
- Real-time Charts — графики в реальном времени

**⚙️ Backend (Django 5.2)**
- REST API — основное API для работы с данными
- WebSocket — для real-time обновлений
- RAG AI — система интеллектуальных ответов

**💾 Database (TimescaleDB)**
- Sensor Data — данные с датчиков
- Analysis Results — результаты анализа
- System Config — конфигурации систем

**🤖 AI/ML (Ollama + Qwen3)**
- LLM Engine — языковая модель
- FAISS Index — векторные индексы
- RAG Pipeline — конвейер обработки

### Потоки данных

1. Сбор данных: Датчики → TimescaleDB
2. Обработка: TimescaleDB → Django AI Engine
3. Анализ: AI Engine → 3 метода диагностики
4. Отображение: Django → WebSocket → Nuxt 4 Frontend
5. Консультации: RAG System → Frontend Chat

---

## 🔧 Методы диагностики

Наша система использует три независимых метода диагностики, которые объединяются в интегральную оценку.

### Метод 1️⃣: Математическое моделирование (вес: 40%)

**Описание**: сравнение реальных показаний с расчетными по математическим моделям.

Формулы:
- Момент насоса: \( M_\text{н} = \frac{V_0 \cdot P}{2\pi \cdot \eta} \)
- Угловая скорость: \( \omega = \frac{M_\text{д} + M_\text{м} \cdot n - M_\text{н}}{I_\text{д} + I_\text{н} + I_{\text{мех}}} \)
- Расход: \( Q = \omega \cdot V_0 \cdot \eta \)

Критерии:
- \(\Delta < 5\%\) → ✅ Норма (score = 0.1)
- \(5\% \le \Delta < 10\%\) → ⚠️ Предупреждение (score = 0.5)
- \(\Delta \ge 10\%\) → 🔴 Неисправность (score = 0.9)

### Метод 2️⃣: Фазовые портреты (вес: 40%)

Описание: анализ траекторий в фазовом пространстве.

- Площадь по Гауссу: \[ S = \tfrac{1}{2} \cdot \left| \sum_{i=1}^{n-1} x_i y_{i+1} + x_n y_1 - \sum_{i=1}^{n-1} x_{i+1} y_i - x_1 y_n \right| \]
- Отклонение: \( \Delta S = \frac{|S - S_{\text{этал}}|}{S_{\text{этал}}} \cdot 100\% \)

Критерии:
- < 10% → Норма (0.1)
- 10–25% → Предотказ (0.5)
- > 25% → Отказ (0.9)

### Метод 3️⃣: Трибодиагностика (вес: 20%)

Описание: лабораторный анализ рабочей жидкости.

Параметры: ISO 4406 (≥4/≥6/≥14 μm), Fe/Cu/Al/Cr/Si, вязкость 40°C, pH, вода ppm, TAN.

Критерии (пример): ISO 15/13/10 — норма; 16/14/11 — внимание; ≥17/15/12 — критика.

---

## 🔄 Процесс диагностики

### Основной цикл

**Шаг 1:** старт диагностики (ручной/авто), окно данных.  
**Шаг 2:** сбор данных: P, Q, T, ω + история TimescaleDB.  
**Шаг 3:** параллельный расчет 3 методов.  
**Шаг 4:** нормализация оценок (0–1).  
**Шаг 5:** интегральная оценка (блочная формула):  
\[ D = 0.4\,M + 0.4\,P + 0.2\,T \]
где \(M\), \(P\), \(T\) — оценки методов.  
**Шаг 6:** статус: D<0.3 — Норма; D<0.6 — Предупреждение; иначе — Неисправность.  
**Шаг 7:** отчет: статусы, метрики, рекомендации, остаточный ресурс.  
**Шаг 8:** запись в БД, уведомления.

Алгоритмы (кратко)
- Матемодель: сбор P/Q/ω → расчет M/ω/Q → ΔP/ΔQ → score.  
- Портреты: x,V,F,P,Q → площадь/ΔS → форма (центр/разрывы/искажения) → score.  
- Трибо: проба 100 мл → частицы/элементы/физ-хим → ISO класс/износ → score.

---

## 💻 Django Models

### Основные сущности
- HydraulicSystem (owner, параметры, активность)
- Equipment (тип, параметры, даты ТО)
- Sensor (тип, диапазон, точность, калибровка)
- SensorData (TimescaleDB, value/raw/quality/flags)

### Диагностические результаты
- MathematicalModelResult (Δ, статус, score)
- PhasePortraitResult (S, ΔS, форма, score)
- TribodiagnosticResult (ISO, элементы, физ-хим, score)
- IntegratedDiagnosticResult (M/P/T/D, статус, alert, рекомендации)

---

## 🖥️ API & Frontend

Эндпоинты:
- POST /api/diagnostics/mathematical-model/
- POST /api/diagnostics/phase-portraits/
- POST /api/diagnostics/tribodiagnostics/
- POST /api/diagnostics/integrated/
- GET  /api/systems/{id}/diagnostic-history/

Nuxt 4 страницы:
- diagnostics/index|mathematical-model|phase-portraits|tribodiagnostics|integrated
- systems/[systemId]/index
- reports/index

Компоненты: DiagnosticChart, PhasePortraitCanvas, TriboAnalysisTable, IntegratedScoreGauge.

---

## 🚀 Roadmap

Sprint 1: модели, миграции, сериализаторы, базовые эндпоинты.  
Sprint 2: алгоритмы 3 методов + интегральная оценка.  
Sprint 3: Nuxt 4 UI, Chart.js, WebSocket, mobile.  
Sprint 4: тесты (unit/integration/E2E), docker, CI/CD.

---

**Версия:** v4.2 FINAL  
**Изменения:** восстановлен полный документ (структура и разделы) + корректная LaTeX-разметка формул без Mermaid и раздела о точках подключения датчиков, как оговорено ранее.