# 🧪 Запуск Pre-commit тестов

## Текущий статус
✅ **Все исправления применены** - Готово к тестированию

---

## 📋 Чеклист исправлений

- [x] **SQL Injection** - Исправлено в `timescale_tasks.py`
- [x] **Import Order** - Исправлено в 11 файлах
- [x] **Code Formatting** - Приведено к стандарту Black
- [x] **Line Length** - Все строки <= 88 символов
- [x] **Missing Imports** - Добавлен `from django.db import models`

---

## 🚀 Команды для тестирования

### 1. Установка pre-commit (если еще не установлен)
```bash
pip install pre-commit
```

### 2. Проверка отдельных инструментов

#### isort (порядок импортов)
```bash
cd backend
python -m isort --check-only --diff .
```

**Ожидаемый результат:** ✅ No issues found

#### black (форматирование)
```bash
cd backend
python -m black --check --diff .
```

**Ожидаемый результат:** ✅ All files would be left unchanged

#### flake8 (линтер)
```bash
cd backend
python -m flake8 .
```

**Ожидаемый результат:** ✅ No errors (или только warnings, которые игнорируются)

#### bandit (безопасность)
```bash
cd backend
python -m bandit -c ../.bandit -r .
```

**Ожидаемый результат:** ✅ No issues found (HIGH severity)

#### mypy (type checking)
```bash
cd backend
python -m mypy --ignore-missing-imports --check-untyped-defs .
```

**Ожидаемый результат:** ⚠️ Warnings допустимы, ошибок быть не должно

---

### 3. Полный запуск pre-commit

```bash
# В корне проекта
pre-commit run --all-files
```

**Ожидаемый вывод:**
```
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
Check Yaml...............................................................Passed
Check for added large files..............................................Passed
Check for case conflicts.................................................Passed
Check for merge conflicts................................................Passed
Check JSON...............................................................Passed
Check Toml...............................................................Passed
Check Xml................................................................Passed
Debug Statements (Python)................................................Passed
Check builtin type constructor use.......................................Passed
Check docstring is first.................................................Passed
Fix requirements.txt.....................................................Passed
Upgrade Django...........................................................Passed
isort....................................................................Passed
black....................................................................Passed
flake8...................................................................Passed
mypy.....................................................................(Passed or warnings)
bandit...................................................................Passed
```

---

## 🔧 Исправление проблем (если они возникнут)

### isort не прошел
```bash
# Автоисправление
cd backend
python -m isort .
```

### black не прошел
```bash
# Автоисправление
cd backend
python -m black .
```

### flake8 ошибки
Проверьте конкретный файл:
```bash
flake8 backend/path/to/file.py --show-source
```

### bandit ошибки
Проверьте конкретную проблему:
```bash
bandit -r backend/path/to/file.py -v
```

---

## 📊 Исправленные файлы

### Критичные (Security)
1. ✅ `backend/apps/diagnostics/timescale_tasks.py`
   - **Проблема:** SQL Injection (B608)
   - **Исправление:** Параметризованные запросы

### Импорты (isort)
1. ✅ `backend/core/settings.py`
2. ✅ `backend/core/celery.py`
3. ✅ `backend/apps/diagnostics/ai_engine.py`
4. ✅ `backend/apps/diagnostics/views.py`
5. ✅ `backend/apps/diagnostics/websocket_consumers.py`
6. ✅ `backend/apps/diagnostics/signals.py`
7. ✅ `backend/apps/diagnostics/timescale_tasks.py`
8. ✅ `backend/apps/rag_assistant/views.py`
9. ✅ `backend/apps/rag_assistant/tasks.py`
10. ✅ `backend/apps/rag_assistant/rag_service.py`
11. ✅ `backend/apps/rag_assistant/management/commands/init_rag_system.py`

### Форматирование (black)
1. ✅ `backend/apps/diagnostics/signals.py`

---

## 🎯 Проверка конкретных исправлений

### SQL Injection (Критично!)
```bash
# Проверить, что в файле нет f-strings в SQL
grep -n "cursor.execute(f" backend/apps/diagnostics/timescale_tasks.py
```
**Ожидаемый результат:** Пустой вывод (no matches)

### Порядок импортов
```bash
# Проверить один из исправленных файлов
head -n 20 backend/core/settings.py
```
**Ожидаемый результат:** 
- `import structlog` в начале файла (строка ~10)
- Нет `# noqa: E402`

### Форматирование длинных строк
```bash
# Проверить signals.py
grep -A 2 "last_reading_at__lt" backend/apps/diagnostics/signals.py
```
**Ожидаемый результат:** Перенос строки после `|` оператора

---

## 📝 Детальная проверка по категориям

### 1. Безопасность (Bandit)
```bash
bandit -ll -r backend/apps/diagnostics/timescale_tasks.py
```

### 2. Импорты (isort)
```bash
isort --check-only backend/core/settings.py
isort --check-only backend/core/celery.py
isort --check-only backend/apps/diagnostics/ai_engine.py
```

### 3. Форматирование (Black)
```bash
black --check backend/apps/diagnostics/signals.py
black --check backend/apps/diagnostics/timescale_tasks.py
```

### 4. Линтер (Flake8)
```bash
flake8 backend/core/settings.py
flake8 backend/apps/diagnostics/timescale_tasks.py
```

---

## ⚡ Быстрый тест (5 минут)

```bash
#!/bin/bash
# Сохраните как test-precommit.sh

echo "🧪 Быстрый тест pre-commit исправлений"
echo "======================================"

cd backend

echo ""
echo "1️⃣ Проверка SQL Injection..."
if grep -q "cursor.execute(f" apps/diagnostics/timescale_tasks.py; then
    echo "❌ ОШИБКА: Найдены f-strings в SQL!"
    exit 1
else
    echo "✅ SQL Injection исправлен"
fi

echo ""
echo "2️⃣ Проверка импортов (isort)..."
python -m isort --check-only . > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Импорты в порядке"
else
    echo "❌ Импорты требуют исправления"
    python -m isort --diff . | head -n 50
fi

echo ""
echo "3️⃣ Проверка форматирования (black)..."
python -m black --check . > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Форматирование в порядке"
else
    echo "⚠️  Форматирование требует корректировки"
fi

echo ""
echo "4️⃣ Проверка линтера (flake8)..."
python -m flake8 . --count
if [ $? -eq 0 ]; then
    echo "✅ Линтер доволен"
else
    echo "⚠️  Есть замечания от линтера"
fi

echo ""
echo "======================================"
echo "✅ Быстрая проверка завершена!"
```

**Запуск:**
```bash
chmod +x test-precommit.sh
./test-precommit.sh
```

---

## 🐛 Отладка проблем

### Проблема: isort находит ошибки
**Решение:**
```bash
# Посмотреть конкретные различия
isort --diff backend/path/to/file.py

# Автоисправить
isort backend/path/to/file.py
```

### Проблема: black находит проблемы
**Решение:**
```bash
# Посмотреть что нужно исправить
black --diff backend/path/to/file.py

# Автоисправить
black backend/path/to/file.py
```

### Проблема: flake8 ошибки E501 (line too long)
**Решение:**
- Проверьте, что строки <= 88 символов
- Используйте Black для автоформатирования
- Если строка в комментарии - разбейте на несколько

### Проблема: bandit находит проблемы
**Решение:**
```bash
# Детальный вывод
bandit -r backend/path/to/file.py -v

# Проверить конкретную проблему
bandit -r backend/path/to/file.py -ll
```

---

## 📈 Метрики успешности

### Минимальные требования
- ✅ Bandit: 0 HIGH severity issues
- ✅ isort: All files pass
- ✅ black: All files formatted
- ⚠️  flake8: 0 errors (warnings допустимы)
- ⚠️  mypy: 0 errors (warnings допустимы)

### Идеальный результат
- ✅ Все pre-commit хуки: Passed
- ✅ 0 ошибок во всех инструментах
- ✅ 0 warnings в bandit
- ✅ 100% code coverage (опционально)

---

## 🎉 После успешного прохождения

```bash
# 1. Зафиксировать изменения
git add backend/

# 2. Коммит
git commit -m "fix: исправлены все проблемы pre-commit

- Исправлен SQL injection в timescale_tasks.py
- Переупорядочены импорты в 11 файлах
- Приведено форматирование к стандарту Black
- Все тесты pre-commit проходят успешно"

# 3. Установить pre-commit hook
pre-commit install

# 4. Проверить, что hook работает
git commit --amend --no-edit
```

---

## 📞 Поддержка

### Если что-то пошло не так:

1. **Проверьте версии инструментов:**
   ```bash
   python --version  # >= 3.10
   black --version   # >= 25.9.0
   isort --version   # >= 7.0.0
   flake8 --version  # >= 7.1.1
   bandit --version  # >= 1.8.3
   ```

2. **Переустановите зависимости:**
   ```bash
   pip install -r requirements-dev.txt --upgrade
   ```

3. **Очистите кэш Python:**
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

4. **Проверьте конфигурации:**
   ```bash
   cat .pre-commit-config.yaml
   cat pyproject.toml
   cat .flake8
   cat .bandit
   ```

---

**Статус:** 🟢 ГОТОВО К ТЕСТИРОВАНИЮ  
**Дата:** 2025-01-XX  
**Версия:** 1.0
