# Docker Infrastructure

## 📦 Структура

```
infrastructure/
├── nginx/              # API Gateway
│   ├── Dockerfile
│   ├── nginx.conf
│   └── conf.d/
├── init-db/            # Database initialization scripts
│   └── 01-init-timescaledb.sql
├── prometheus/         # Monitoring (optional)
└── grafana/            # Dashboards (optional)
```

## 🚀 Быстрый старт

### 1. Очистка старых конфигураций
```bash
# Linux/Mac
bash cleanup_docker.sh

# Windows
cleanup_docker.bat
```

### 2. Настройка окружения
```bash
cp .env.example .env
# Отредактируй .env (пароли, ключи, порты)
```

### 3. Запуск
```bash
# Production
docker-compose up --build -d

# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Или через Makefile
make build
make up
```

## 📊 Endpoints

После запуска доступны:

| Service | URL | Description |
|---------|-----|-------------|
| FastAPI Backend | http://localhost:8100 | Core API |
| FastAPI Docs | http://localhost:8100/docs | OpenAPI UI |
| Django Admin | http://localhost:8000/admin | Admin Panel |
| GNN Service | http://localhost:8001 (internal) | ML Inference |
| PostgreSQL | localhost:5432 | Database |
| Redis | localhost:6379 | Cache/Queue |

## 🛠️ Makefile команды

```bash
make help              # Показать все команды
make build             # Собрать контейнеры
make up                # Запустить сервисы
make down              # Остановить сервисы
make logs              # Просмотр логов
make logs SERVICE=gnn_service  # Логи конкретного сервиса
make migrate           # Выполнить миграции
make test              # Запустить тесты
make health            # Проверить health endpoints
make backup-db         # Бэкап базы данных
```

## 🔒 Security Checklist (Production)

- [ ] Изменить все пароли в `.env`
- [ ] Сгенерировать надёжные SECRET_KEY
- [ ] Включить HTTPS (SSL certificates)
- [ ] Настроить `internal: true` для internal network
- [ ] Включить rate limiting в Nginx
- [ ] Настроить firewall (только 80/443 порты)
- [ ] Включить мониторинг (Prometheus + Grafana)
- [ ] Настроить backups (автоматические)
- [ ] Логирование в centraliz

ed system (ELK)

## 📝 Миграция с старой конфигурации

1. Бэкап базы данных:
```bash
docker exec hdx-postgres pg_dump -U user hydraulic_db > backup.sql
```

2. Остановить старые контейнеры:
```bash
docker-compose down -v
```

3. Очистка:
```bash
bash cleanup_docker.sh
```

4. Запуск новой конфигурации:
```bash
docker-compose up --build -d
```

5. Восстановление данных:
```bash
cat backup.sql | docker exec -i hdx-postgres psql -U user hydraulic_db
```

## 🐛 Troubleshooting

### Port already in use
```bash
# Найти процесс
lsof -i :8100
# Или изменить порт в .env
```

### Cannot connect to database
```bash
# Проверить логи
docker-compose logs postgres
# Пересоздать контейнер
docker-compose down -v
docker-compose up postgres
```

### Permission denied
```bash
# Дать права на volumes
sudo chown -R 1000:1000 ./services
```
