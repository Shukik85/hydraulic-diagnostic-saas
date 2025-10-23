# 🪟 Windows Setup Guide

Комплексное руководство по настройке проекта Hydraulic Diagnostic SaaS на Windows.

## 📋 Предварительные требования

### Обязательные инструменты

1. **Python 3.11+**
   ```powershell
   # Проверка версии
   python --version
   
   # Если Python не установлен, скачайте с python.org
   # Убедитесь, что добавили Python в PATH
   ```

2. **Node.js 18+**
   ```powershell
   # Проверка версии
   node --version
   npm --version
   
   # Установка через winget (рекомендуется)
   winget install OpenJS.NodeJS
   ```

3. **Git**
   ```powershell
   # Проверка
   git --version
   
   # Установка через winget
   winget install Git.Git
   ```

4. **Docker Desktop** (опционально, для контейнеризации)
   ```powershell
   # Скачать с docker.com или через winget
   winget install Docker.DockerDesktop
   ```

### Рекомендуемые инструменты

1. **uv** - быстрый Python package manager
   ```powershell
   # Установка через pip
   pip install uv
   
   # Или через PowerShell
   irm https://astral.sh/uv/install.ps1 | iex
   ```

2. **Windows Terminal** - современный терминал
   ```powershell
   # Установка из Microsoft Store или
   winget install Microsoft.WindowsTerminal
   ```

3. **PowerShell 7+**
   ```powershell
   winget install Microsoft.PowerShell
   ```

## 🚀 Быстрый старт

### 1. Клонирование репозитория
```powershell
git clone https://github.com/Shukik85/hydraulic-diagnostic-saas.git
cd hydraulic-diagnostic-saas
```

### 2. Настройка среды разработки

#### Вариант А: С использованием PowerShell скрипта (рекомендуется)
```powershell
# Установка зависимостей
.\make.ps1 install-dev

# Запуск в режиме разработки
.\make.ps1 dev
```

#### Вариант Б: Ручная настройка

**Backend:**
```powershell
# Создание виртуальной среды
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1

# Установка зависимостей
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Применение миграций (требуется база данных)
python manage.py migrate

# Создание суперпользователя
python manage.py createsuperuser

# Запуск сервера разработки
python manage.py runserver
```

**Frontend:**
```powershell
cd nuxt_frontend
npm install
npm run dev
```

## 🔧 Использование PowerShell скрипта

Вместо `make` команд используйте PowerShell скрипт:

```powershell
# Показать все доступные команды
.\make.ps1 help

# Основные команды разработки
.\make.ps1 install-dev     # Установка зависимостей
.\make.ps1 dev             # Запуск с Docker
.\make.ps1 dev-local       # Запуск без Docker
.\make.ps1 stop            # Остановка сервисов

# Тестирование
.\make.ps1 test            # Все тесты
.\make.ps1 test-backend    # Backend тесты
.\make.ps1 test-frontend   # Frontend тесты
.\make.ps1 test-coverage   # Тесты с покрытием

# Качество кода
.\make.ps1 lint-backend    # Линтинг backend
.\make.ps1 lint-frontend   # Линтинг frontend
.\make.ps1 format-backend  # Форматирование backend
.\make.ps1 format-frontend # Форматирование frontend
.\make.ps1 pre-commit      # Pre-commit хуки

# Информация
.\make.ps1 status          # Статус проекта
.\make.ps1 urls            # URL приложений
.\make.ps1 clean           # Очистка артефактов
```

## 🛠️ Настройка инструментов разработки

### Pre-commit hooks
```powershell
# Установка pre-commit
pip install pre-commit

# Установка хуков
pre-commit install

# Запуск на всех файлах
pre-commit run --all-files
```

### Ruff (современный линтер)
```powershell
# Установка
pip install ruff

# Использование
ruff check backend/
ruff format backend/
```

### Настройка IDE

#### VS Code
1. Установите расширения:
   - Python
   - Pylance
   - Ruff
   - Vue Language Features (Vetur)
   - ESLint
   - Prettier

2. Настройка settings.json:
```json
{
    "python.defaultInterpreterPath": "./backend/.venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm
1. Настройте интерпретатор Python: `backend\.venv\Scripts\python.exe`
2. Включите Ruff в Settings > Tools > External Tools
3. Настройте форматирование с Black

## 📊 База данных

### Локальная разработка

#### SQLite (по умолчанию)
Никакой дополнительной настройки не требуется.

#### PostgreSQL + TimescaleDB (рекомендуется)

**Вариант 1: Docker (рекомендуется)**
```powershell
# Запуск через docker-compose
docker-compose -f docker-compose.dev.yml up -d postgres
```

**Вариант 2: Локальная установка**
1. Установите PostgreSQL 16+
2. Установите TimescaleDB extension
3. Обновите настройки в `.env`:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/hydraulic_diagnostic
```

### Миграции
```powershell
cd backend

# Создание миграций
python manage.py makemigrations

# Применение миграций
python manage.py migrate

# Откат миграций
python manage.py migrate app_name 0001
```

## 🧪 Тестирование

### Backend тесты
```powershell
cd backend

# Все тесты
pytest

# С покрытием
pytest --cov=apps --cov-report=html

# Конкретный модуль
pytest apps/diagnostics/tests/

# Параллельные тесты (быстрее)
pytest -n auto

# Только быстрые тесты
pytest -m "not slow"
```

### Frontend тесты
```powershell
cd nuxt_frontend

# Все тесты
npm run test

# С покрытием
npm run test:coverage

# В режиме наблюдения
npm run test:watch

# UI тесты
npm run test:ui
```

## 🔍 Отладка

### Backend отладка

**Django Debug Toolbar:**
Включен в development режиме по умолчанию.

**iPDB отладчик:**
```python
# В коде
import ipdb; ipdb.set_trace()
```

**Логирование:**
```python
import logging
logger = logging.getLogger(__name__)
logger.debug("Debug message")
```

### Frontend отладка

**Vue DevTools:**
Установите браузерное расширение Vue.js devtools.

**Console logging:**
```javascript
console.log('Debug info:', data)
console.table(arrayData)
```

## ❗ Решение проблем

### Частые проблемы и решения

#### 1. Python/pip проблемы
```powershell
# Обновление pip
python -m pip install --upgrade pip

# Переустановка пакета
pip uninstall package_name
pip install package_name

# Очистка кэша pip
pip cache purge
```

#### 2. Node.js/npm проблемы
```powershell
# Очистка npm кэша
npm cache clean --force

# Переустановка node_modules
Remove-Item node_modules -Recurse -Force
Remove-Item package-lock.json
npm install
```

#### 3. Проблемы с правами доступа
```powershell
# Запуск PowerShell от имени администратора
# Или изменение политики выполнения скриптов
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 4. Ошибки кодировки
```powershell
# Установка UTF-8 по умолчанию
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")

# Или в PowerShell профиле
$env:PYTHONIOENCODING="utf-8"
```

#### 5. Django settings проблемы
Убедитесь, что используется правильный путь:
```python
# В pytest.ini или pyproject.toml
DJANGO_SETTINGS_MODULE = "core.settings"
# НЕ "backend.core.settings.base"
```

#### 6. Import ошибки
```powershell
# Проверьте PYTHONPATH
$env:PYTHONPATH=".;./backend"

# Или добавьте в .env файл
PYTHONPATH=.;./backend
```

### Полезные команды диагностики

```powershell
# Информация о Python
python -m site
python -c "import sys; print('\n'.join(sys.path))"

# Установленные пакеты
pip list
pip show package_name

# Информация о системе
systeminfo | findstr /C:"OS"
$PSVersionTable

# Процессы
Get-Process python
Get-Process node

# Порты
netstat -ano | findstr :8000
netstat -ano | findstr :3000
```

## 🚀 Продвинутые возможности

### Docker Development
```powershell
# Сборка образов
docker-compose -f docker-compose.dev.yml build

# Запуск отдельных сервисов
docker-compose -f docker-compose.dev.yml up postgres redis

# Логи сервисов
docker-compose -f docker-compose.dev.yml logs -f

# Выполнение команд в контейнере
docker-compose -f docker-compose.dev.yml exec backend python manage.py shell
```

### Performance Monitoring
```powershell
# Установка и запуск профайлера
pip install py-spy
py-spy top --pid $(Get-Process python).Id

# Django Debug Toolbar для веб-профилирования
# Доступен по адресу http://localhost:8000
```

### CI/CD локально
```powershell
# Запуск act (локальный GitHub Actions)
winget install nektos.act
act -j backend-test
```

## 📚 Дополнительные ресурсы

- [Django Documentation](https://docs.djangoproject.com/)
- [Nuxt 3 Documentation](https://nuxt.com/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Docker Desktop for Windows](https://docs.docker.com/desktop/windows/)
- [PowerShell Documentation](https://docs.microsoft.com/en-us/powershell/)

## 💡 Полезные советы

1. **Используйте Windows Terminal** с PowerShell 7+ для лучшего опыта
2. **Настройте WSL2** для Linux-совместимости (опционально)
3. **Используйте VS Code** с Remote-Containers для изолированной разработки
4. **Включите Windows Developer Mode** для символических ссылок
5. **Регулярно обновляйте зависимости** с помощью `pip-tools` или `uv`

---

**Нужна помощь?** Создайте issue в репозитории или обратитесь к команде разработки.
