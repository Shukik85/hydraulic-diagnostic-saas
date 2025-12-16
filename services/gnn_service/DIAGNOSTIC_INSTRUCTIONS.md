# 🔍 Локальная Диагностика - Пошаговая Инструкция

**Дата**: December 16, 2025  
**Статус**: Подробные инструкции для анализа ошибок локально

---

## 📋 Что Включает Диагностика

Скрипт `LOCAL_DIAGNOSTIC.sh` проверяет:

1. **pytest** - Запуск тестов
2. **mypy** - Проверка типов (strict mode)
3. **ruff** - Linting и code quality
4. **coverage** - Покрытие тестами

---

## 🚀 Быстрый Старт (5 минут)

### Шаг 1: Убедитесь что находитесь в правильной директории

```bash
cd /path/to/hydraulic-diagnostic-saas/services/gnn_service

# Проверьте что вы видите файлы
ls -la | grep -E 'src|tests|LOCAL_DIAGNOSTIC'
```

### Шаг 2: Установите зависимости

```bash
# Активируйте виртуальное окружение
source .venv/Scripts/activate  # Windows Git Bash
# или
.venv\Scripts\activate  # Windows cmd

# Установите инструменты диагностики
pip install pytest pytest-asyncio pytest-cov mypy ruff

# Проверьте что все установилось
pytest --version
mypy --version
ruff --version
```

### Шаг 3: Запустите диагностику

```bash
# Сделайте скрипт исполняемым (только Linux/Mac)
chmod +x LOCAL_DIAGNOSTIC.sh

# Запустите диагностику
bash LOCAL_DIAGNOSTIC.sh

# Или на Windows PowerShell:
bash ./LOCAL_DIAGNOSTIC.sh
```

---

## 📊 Что Вы Увидите

Вывод будет выглядеть примерно так:

```
╔════════════════════════════════════════════════════════════════╗
║     🔍 GNN SERVICE - LOCAL DIAGNOSTIC REPORT                   ║
║     2025-12-16 15:45:30                                        ║
╚════════════════════════════════════════════════════════════════╝

1️⃣  PYTEST - Test Suite
────────────────────────────────────────────────────────────────
✅ pytest найден

Running: pytest tests/ -v --tb=short --no-header
────────────────────────────────────────────────────────────────

Примеры:
  ✅ tests/test_topology_service.py::TestTopologyService::test_singleton_pattern PASSED
  ✅ tests/test_normalizer.py::TestNormalizer::test_normalize_values PASSED
  ❌ tests/test_dynamic_edges_integration.py::TestEndToEndPipeline::test_graph_construction_14d_edges ERROR
     ↳ pydantic_core._pydantic_core.ValidationError: 4 validation errors for ComponentSpec

2️⃣  MYPY - Type Checking (Strict Mode)
────────────────────────────────────────────────────────────────
✅ mypy найден

Running: mypy src/ --strict --no-incremental
────────────────────────────────────────────────────────────────

ОШИБКИ ПО ФАЙЛАМ:
  12 errors: src/data/feature_engineer.py
  18 errors: src/data/graph_builder.py
  20 errors: src/models/layers.py
  18 errors: src/training/trainer.py
   6 errors: src/api/main.py

ОШИБКИ ПО ТИПАМ:
   8 [no-any-return]
   6 [arg-type]
   5 [assignment]
   4 [operator]

ПЕРВЫЕ 20 ОШИБОК:
src/data/feature_engineer.py:154: error: Returning Any from function declared to return "ndarray[tuple[Any, ...], dtype[Any]]" [no-any-return]
src/data/graph_builder.py:213: error: Statement is unreachable [unreachable]
...

📊 ИТОГО: 75 mypy ошибок

3️⃣  RUFF - Linting & Code Quality
────────────────────────────────────────────────────────────────
✅ ruff найден

Running: ruff check src/ tests/
────────────────────────────────────────────────────────────────

НАРУШЕНИЯ ПО ПРАВИЛАМ:
   6 E501  (line too long)
   3 F401  (unused import)
   2 F841  (unused variable)

📊 ИТОГО: ~11 ruff проблем

4️⃣  COVERAGE - Test Coverage
────────────────────────────────────────────────────────────────
...
```

---

## 🎯 Интерпретация Результатов

### ✅ Если Все Зеленое

```
Status:    Test Suite       mypy        Ruff
           ✅ PASS         ✅ PASS     ✅ PASS
```

**Значит**: Код полностью готов к production! 🚀

---

### ❌ Если Есть PYTEST ошибки

**Вероятные ошибки**:

```python
pydantic_core._pydantic_core.ValidationError: 4 validation errors for ComponentSpec
  sensors: Field required [type=missing, ...]
  feature_dim: Field required [type=missing, ...]
  nominal_pressure_bar: Field required [type=missing, ...]
  nominal_flow_lpm: Field required [type=missing, ...]
```

**Причина**: Test fixtures используют старую схему ComponentSpec

**Серьезность**: 🟡 **СРЕДНЯЯ** - не влияет на production код

**Фиксить**: Файл `tests/test_dynamic_edges_integration.py` (~30 минут)

---

### ⚠️ Если Есть MYPY ошибки

**Типичные ошибки**:

```
src/data/feature_engineer.py:154: error: Returning Any from function [no-any-return]
src/models/layers.py:128: error: Returning Any from function [no-any-return]
src/training/trainer.py:305: error: Incompatible type [arg-type]
```

**Классификация**:

| Файл | Ошибок | Серьезность | Production? |
|------|--------|-------------|-------------|
| `src/data/feature_engineer.py` | 12 | 🟡 LOW | Нет (legacy) |
| `src/data/graph_builder.py` | 18 | 🟡 LOW | Нет (training) |
| `src/models/layers.py` | 20 | 🟡 MED | Частично (работает) |
| `src/training/trainer.py` | 18 | 🟡 MED | Нет (training) |
| `src/api/main.py` | 6 | 🔴 HIGH | ДА! |

**Основной вывод**: Большинство ошибок в non-critical коде ✅

---

### 🟢 Если RUFF не показывает ошибок

**Хорошо!** Code style в норме ✅

---

## 🔧 Ручные Проверки (Если скрипт не работает)

### Проверка 1: Только тесты

```bash
pytest tests/ -v --tb=short

# Или только specific test
pytest tests/test_dynamic_edges_integration.py::TestEndToEndPipeline::test_graph_construction_14d_edges -v
```

---

### Проверка 2: Только mypy

```bash
# Все файлы strict
mypy src/ --strict

# Только критичные файлы
mypy src/api/main.py src/models/gnn_model.py src/inference/ --strict

# С более подробным выводом
mypy src/ --strict --show-error-codes --show-error-context
```

---

### Проверка 3: Только ruff

```bash
# Только checks (без fixes)
ruff check src/ tests/

# Показать конкретные правила
ruff check src/ --select E501,F401

# Автоматический fix (ОСТОРОЖНО!)
ruff check src/ --fix
```

---

### Проверка 4: Test coverage

```bash
pytest tests/ --cov=src --cov-report=html

# Откройте htmlcov/index.html в браузере
open htmlcov/index.html  # Mac
start htmlcov/index.html # Windows
```

---

## 📈 Результаты Хранятся в

```
/tmp/pytest_output.txt   ← Full pytest output
/tmp/mypy_output.txt     ← Full mypy output
/tmp/ruff_output.txt     ← Full ruff output
```

**Смотрите полный вывод:**

```bash
cat /tmp/pytest_output.txt
cat /tmp/mypy_output.txt
cat /tmp/ruff_output.txt
```

---

## 🎯 Интерпретация Каждой Категории Ошибок

### ТЕСТЫ (pytest)

**ЗЕЛЕНЫЕ** = ✅ Все работает

**КРАСНЫЕ** = ❌ Есть проблемы (обычно schema mismatch)

**Тип проблемы**: ValidationError при создании test fixtures

**Влияние**: На production код НЕ влияет ✅

---

### ТИПЫ (mypy)

**0 ошибок** = ✅ Perfect type safety

**< 10 ошибок в critical коде** = 🟢 OK для production

**> 50 ошибок** = 🟡 Есть non-critical legacy код

**Типичное для нас**: 75 ошибок, но большинство в non-critical коде

---

### ЛINTING (ruff)

**0 ошибок** = ✅ Perfect code style

**< 20 ошибок** = 🟢 Acceptable

**E501** (line too long) = 🟡 Cosmetici, ignorable

**F401** (unused import) = 🟡 Minor

---

## 📋 Чеклист Интерпретации

- [ ] Запустить диагностику
- [ ] Прочитать вывод PYTEST
  - [ ] Есть ошибки?
  - [ ] Какие файлы затронуты?
- [ ] Прочитать вывод MYPY
  - [ ] Сколько ошибок всего?
  - [ ] Какие файлы?
  - [ ] Какие типы ошибок (по коду)?
- [ ] Прочитать вывод RUFF
  - [ ] Есть критичные violations?
  - [ ] Что можно auto-fix?
- [ ] Определить серьезность
  - [ ] Критичные для production?
  - [ ] Или это non-critical legacy?
- [ ] Решить на стратегию фиксинга
  - [ ] Deploy now?
  - [ ] Phase 2A (2 часа)?
  - [ ] Phase 2B (6 часов)?

---

## 🆘 Проблемы с Запуском Скрипта

### Проблема: "command not found: bash"

**Решение** (Windows):
```bash
scoop install git  # или используйте Git Bash
```

### Проблема: pytest не найден

**Решение**:
```bash
pip install pytest pytest-asyncio pytest-cov
```

### Проблема: Permission denied (Linux/Mac)

**Решение**:
```bash
chmod +x LOCAL_DIAGNOSTIC.sh
bash LOCAL_DIAGNOSTIC.sh
```

### Проблема: Скрипт медленный

**Нормально!** Первый запуск может быть медленным (~2-3 минуты). Последующие быстрее.

---

## ✨ Что Дальше

После того как запустили диагностику:

1. **Изучите результаты** - Какие ошибки, где, почему
2. **Дайте мне знать** - Поделитесь выводом
3. **Решим стратегию** - Quick Fix / Phase 2A / Phase 2B / Deploy Now
4. **Я помогу** - С исправлениями конкретных ошибок

---

## 📞 Если Вам Нужна Помощь

**Скопируйте и поделитесь:**

```bash
# Вывод полного диагностики
bash LOCAL_DIAGNOSTIC.sh 2>&1 | head -100

# Или отдельные проверки
pytest tests/ -v --tb=line 2>&1 | head -50
mypy src/ --strict 2>&1 | head -50
```

---

**Готовы начать?** 🚀

Запустите диагностику и дайте мне знать результаты!
