# 🚀 Quick Rebuild Guide

## 🔧 Исправлена ошибка psycopg[pool]!

### ❓ Проблема
```
django.core.exceptions.ImproperlyConfigured: Error loading psycopg_pool module.
Did you install psycopg[pool]?
```

### ✅ Решение
Добавлен `psycopg[pool]` в requirements.txt!

## 🚀 Быстрая пересборка

### Option 1: PowerShell Script (Recommended)
```powershell
# Останови текущие контейнеры
docker-compose down -v

# Пересобери (будет быстро благодаря кешу!)
docker-compose build

# Запусти
docker-compose up -d

# Проверь логи
docker-compose logs backend --tail=50 -f
```

### Option 2: Автоматический скрипт
```powershell
# Используй обновлённый скрипт
.\fix-docker-issues.ps1
```

### Option 3: Batch Script
```cmd
docker-quick-start.bat
```

## ⏱️ Ожидаемое время

| Действие | Время (с кешем) |
|---------|-------------------|
| 🔨 Первая сборка | 5-10 мин |
| ⚡ Последующие | 1-2 мин |
| 🚀 Изменения кода | 30 сек |

## 🔍 Проверка результата

### 1. Проверь статус контейнеров
```bash
docker-compose ps
```

Ожидаемый результат:
```
NAME              STATUS
hdx-backend       Up (healthy)
hdx-postgres      Up (healthy)
hdx-redis         Up (healthy)
hdx-celery        Up (healthy)
hdx-celery-beat   Up (healthy)
```

### 2. Проверь backend health
```bash
curl http://localhost:8000/health/
```

Или открой в браузере: http://localhost:8000/health/

### 3. Проверь Django Admin
Открой: http://localhost:8000/admin

🔑 **Login**: admin / admin123

### 4. Проверь API Documentation
Открой: http://localhost:8000/api/schema/swagger-ui/

## 🐛 Troubleshooting

### Проблема: Контейнер не запускается
```bash
# Смотри логи
docker-compose logs backend --tail=100
```

### Проблема: psycopg_pool ошибка осталась
```bash
# Полная пересборка без кеша
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Проблема: База данных не доступна
```bash
# Проверь PostgreSQL
docker-compose logs db --tail=50

# Перезапусти DB
docker-compose restart db
```

### Проблема: Миграции не применились
```bash
# Запусти миграции вручную
docker-compose exec backend python manage.py migrate
```

## ⚡ Полезные команды

```bash
# Перестроить только backend
docker-compose build backend

# Перезапустить backend
docker-compose restart backend

# Посмотреть логи в реальном времени
docker-compose logs -f

# Зайти в bash backend контейнера
docker-compose exec backend bash

# Проверить Django
docker-compose exec backend python manage.py check

# Создать суперюзера
docker-compose exec backend python manage.py createsuperuser
```

## 💾 Кеш информация

```bash
# Проверить размер кеша
docker system df

# Подробная информация
docker builder du
```

## 🎉 Что должно работать

После пересборки:

- ✅ **Backend API**: http://localhost:8000
- ✅ **Django Admin**: http://localhost:8000/admin (admin/admin123)
- ✅ **API Docs**: http://localhost:8000/api/schema/swagger-ui/
- ✅ **PostgreSQL**: localhost:5432
- ✅ **Redis**: localhost:6379
- ✅ **Нет psycopg[pool] ошибок**
- ✅ **Все миграции применены**
- ✅ **Health check проходит**

---

🚀 **Благодаря pip кешу, пересборка займёт всего 1-2 минуты вместо 40!**