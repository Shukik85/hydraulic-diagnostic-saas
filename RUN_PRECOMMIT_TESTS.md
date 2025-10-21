# 🧪 Запуск и проверка pre-commit хуков

Этот документ описывает, как локально запускать и проверять pre-commit хуки в проекте Hydraulic Diagnostic SaaS и быстро устранять типовые замечания.

---

## 📦 Требования
- Python 3.10+
- Установленные зависимости разработки: `pip install -r backend/requirements-dev.txt`
- Установленный pre-commit: `pip install pre-commit`

Инструменты и их конфигурация берутся из `.pre-commit-config.yaml` и `pyproject.toml` (Black, isort, flake8, mypy, bandit, форматтеры и проверки файлов).

---

## 🚀 Быстрый старт

```bash
# Установка хуков в git
pre-commit install

# Полный прогон всех хуков по всем файлам
pre-commit run --all-files

# Проверка качества через Makefile (если доступно)
make check || true
```

Ожидаемый результат: все проверки Passed. Если есть замечания — используйте подсказки из раздела «Быстрое исправление».  

---

## 🔍 Точечные проверки

```bash
# Импорты (isort)
python -m isort --check-only --diff backend/

# Форматирование (black)
python -m black --check --diff backend/

# Линтер (flake8)
python -m flake8 backend/

# Безопасность (bandit)
python -m bandit -c .bandit -r backend/

# Типы (mypy)
python -m mypy --ignore-missing-imports --check-untyped-defs backend/
```

---

## 🧰 Быстрое исправление типовых замечаний

### Импорты и форматирование
```bash
isort backend/
black backend/
```

### Flake8: часто встречающиеся правила
- E226 — пробелы вокруг арифметических операторов: `a + b * c`
- E402 — импорты должны быть вверху файла. Допустимо добавлять `# noqa: E402` для импортов после `django.setup()` в скриптах.
- E704 — не размещать `def` и тело на одной строке
- F841 — удалить/использовать неиспользуемые переменные
- C901 — чрезмерная сложность: упростить или временно пометить `# noqa: C901` при необходимости и обосновании
- W291/W292/W293 — хвостовые пробелы / пустая строка в конце файла / пустые строки с пробелами

### Bandit (безопасность)
- Исключить f-strings в SQL. Всегда использовать параметризованные запросы:
```python
# Плохо
cursor.execute(f"SELECT compress_chunk('{full_name}')")

# Хорошо
cursor.execute("SELECT compress_chunk(%s)", [full_name])
```

### Mypy
- Добавлять аннотации типов в публичные функции
- При необходимости локально игнорировать сложные участки `# type: ignore[<code>]`

---

## 🧪 Полезные сценарии

```bash
# Полный прогон и автоисправление форматирования
isort backend/ && black backend/

# Прогон только критичных проверок
flake8 backend/ && bandit -c .bandit -r backend/

# Комбинированный прогон (быстрый)
pre-commit run trailing-whitespace -a && \
pre-commit run end-of-file-fixer -a && \
python -m isort . && python -m black . && \
python -m flake8 backend/
```

---

## 🧭 Отчётность и CI
- Все хуки запускаются в GitHub Actions в workflow `main.yml/ci.yml` в части линтинга и статических проверок.
- При падении любой проверки в CI ориентируйтесь на этот документ и сообщения из логов Actions.

---

## ❓ FAQ
- В скриптах, где требуется `django.setup()`, импорты Django-зависимых модулей допустимы после setup с комментарием `# noqa: E402`.
- Для слишком больших функций добавляйте `# noqa: C901` только по необходимости и с задачей на рефакторинг.
- Для временного отключения правила на строку используйте `# noqa: <CODE>` и оставляйте пояснение в комментарии.

---

Обновлено: 2025-10-21
