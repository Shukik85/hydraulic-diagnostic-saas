# 🚀 Docker Deployment Guide - Hydraulic Diagnostic SaaS

## 📋 Предварительные требования

### 1. Установить Docker Desktop
- **Windows**: https://docs.docker.com/desktop/install/windows-install/
- **Минимум**: 16GB RAM, 50GB свободного места

### 2. Установить Docker Compose
```bash
# Проверить версию
docker-compose --version
```

---

## 🔧 Шаг 1: Настройка окружения

### Создать .env файл:
```bash
# Скопировать example
cp .env.example .env

# Отредактировать значения
nano .env  # или notepad .env
```

### Обязательно изменить:
```env
POSTGRES_PASSWORD=your_secure_password
DJANGO_SECRET_KEY=random_50_char_string
REDIS_PASSWORD=another_secure_password
RAG_ADMIN_KEY=secure_admin_key
GRAFANA_PASSWORD=grafana_admin_pass
```

---

## 🐋 Шаг 2: Запуск всего стека

### Вариант А: Production (все сервисы)
```bash
# Собрать все образы
docker-compose build

# Запустить в фоне
docker-compose up -d

# Проверить статус
docker-compose ps
```

### Вариант Б: Development (выборочно)
```bash
# Только база данных и Redis
docker-compose up -d postgres redis

# Backend
docker-compose up -d backend

# Frontend
docker-compose up -d frontend
```

### Вариант В: С Ollama
```bash
# Запустить с LLM
docker-compose up -d ollama rag_service

# Скачать модель в Ollama
docker-compose exec ollama ollama pull deepseek-r1:1.5b
```

---

## 🎯 Шаг 3: Инициализация

### 1. Django migrations:
```bash
docker-compose exec backend python manage.py migrate
docker-compose exec backend python manage.py createsuperuser
```

### 2. Загрузить модель GNN:
```bash
docker-compose exec gnn_service python -c "from inference.engine import load_model; load_model()"
```

### 3. Индексировать KB для RAG:
```bash
docker-compose exec rag_service python -c "from knowledge_base import index_documents; index_documents()"
```

---

## ✅ Шаг 4: Проверка работы

### Health checks:
```bash
# Backend
curl http://localhost:8000/health/

# GNN Service
curl http://localhost:8001/health

# RAG Service
curl http://localhost:8004/health

# Frontend
curl http://localhost:3000
```

### Открыть в браузере:
- **Frontend**: http://localhost:3000
- **Backend Admin**: http://localhost:8000/admin
- **Grafana**: http://localhost:3001 (admin / grafana_password)
- **Prometheus**: http://localhost:9090

---

## 📊 Мониторинг

### Логи:
```bash
# Все сервисы
docker-compose logs -f

# Конкретный сервис
docker-compose logs -f backend
docker-compose logs -f rag_service

# Последние 100 строк
docker-compose logs --tail=100 gnn_service
```

### Метрики:
```bash
# Использование ресурсов
docker stats

# Список контейнеров
docker-compose ps
```

---

## 🔄 Обновление

### Обновить код:
```bash
# Pull changes
git pull origin master

# Rebuild and restart
docker-compose build
docker-compose up -d
```

### Обновить только один сервис:
```bash
docker-compose build backend
docker-compose up -d backend
```

---

## 🛑 Остановка и очистка

### Остановить все:
```bash
docker-compose down
```

### Остановить с удалением volumes:
```bash
docker-compose down -v
```

### Очистить всё (осторожно!):
```bash
docker-compose down -v --rmi all
docker system prune -a --volumes
```

---

## 🐛 Troubleshooting

### Контейнер не стартует:
```bash
# Проверить логи
docker-compose logs backend

# Войти в контейнер
docker-compose exec backend bash
```

### Ollama не отвечает:
```bash
# Перезапустить
docker-compose restart ollama

# Проверить модели
docker-compose exec ollama ollama list
```

### База данных не доступна:
```bash
# Проверить Postgres
docker-compose exec postgres psql -U hydraulic_user -d hydraulic_db

# Пересоздать volume
docker-compose down -v postgres
docker-compose up -d postgres
```

### Порт занят:
```bash
# Найти процесс
netstat -ano | findstr :3000

# Убить процесс (Windows)
taskkill /PID <PID> /F

# Изменить порт в docker-compose.yml
ports:
  - "3001:3000"  # Внешний порт изменён
```

---

## 🚀 Production Deployment

### 1. Использовать .env для production
```bash
cp .env.example .env.production
# Настроить production значения
```

### 2. Включить SSL/TLS
```yaml
# Добавить Nginx/Traefik reverse proxy
nginx:
  image: nginx:alpine
  ports:
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/ssl
```

### 3. Backup базы данных
```bash
# Создать backup
docker-compose exec postgres pg_dump -U hydraulic_user hydraulic_db > backup.sql

# Восстановить
docker-compose exec -T postgres psql -U hydraulic_user hydraulic_db < backup.sql
```

---

## 📦 Полезные команды

```bash
# Перезапустить сервис
docker-compose restart backend

# Войти в контейнер
docker-compose exec backend bash

# Выполнить Django команду
docker-compose exec backend python manage.py <command>

# Просмотр использования CPU/RAM
docker stats --no-stream

# Очистить unused images
docker image prune -a

# Экспорт образа
docker save hydraulic-backend -o backend.tar

# Импорт образа
docker load -i backend.tar
```

---

## ✅ Готово!

Теперь у тебя запущен полный production-ready стек! 🎉

**Следующие шаги:**
1. ✅ Настрой .env с реальными паролями
2. ✅ Запусти `docker-compose up -d`
3. ✅ Создай superuser в Django
4. ✅ Открой http://localhost:3000
5. ✅ Подавай заявку в акселератор! 🚀
