# API Documentation

## 📚 Overview

Централизованная OpenAPI спецификация для всех сервисов Hydraulic Diagnostic SaaS.

## 📂 Структура

```
docs/api/
├── openapi.yaml              # Агрегированная спецификация (все сервисы)
└── README.md                 # Эта документация

services/
├── backend/
│   └── openapi.yaml          # Backend API endpoints
├── gnn_service/
│   └── openapi.yaml          # GNN inference endpoints
└── rag_service/
    └── openapi.yaml          # RAG interpretation endpoints (future)
```

## 🔧 Использование

### 1. Обновление спецификации

После изменения любого `services/*/openapi.yaml`:

```bash
python tools/aggregate_openapi.py
```

Это создаст/обновит `docs/api/openapi.yaml` с полной спецификацией.

### 2. Генерация Frontend клиентов

```bash
bash tools/generate_frontend_clients.sh
```

Создаст TypeScript клиенты в `services/frontend/api/generated/`

### 3. Просмотр документации

Swagger UI (локально):
```bash
docker run -p 8080:8080 -e SWAGGER_JSON=/foo/openapi.yaml -v $(pwd)/docs/api:/foo swaggerapi/swagger-ui
```

Откройте: http://localhost:8080

## 🚀 CI/CD Integration

Файл `.github/workflows/api-docs.yml` автоматически:
- Агрегирует спецификации при изменении
- Генерирует клиенты для frontend
- Коммитит изменения

## 📝 Best Practices

1. **Каждый сервис** имеет свою `openapi.yaml`
2. **Не редактируйте** `docs/api/openapi.yaml` вручную (генерируется автоматически)
3. **Версионируйте API** через semantic versioning
4. **Тестируйте** контракты после изменений

## 🔗 Endpoints

### Backend API
- `POST /metadata/save` - Сохранение метаданных системы
- `GET /metadata/{user_id}/{system_id}` - Получение метаданных
- `POST /sensor/ingest` - Ingestion sensor data

### GNN Service
- `POST /gnn/infer` - Universal GNN inference
- `GET /gnn/health` - Health check

### RAG Service (future)
- `POST /rag/interpret` - Интерпретация аномалий
- `GET /rag/report` - Генерация отчётов

## 📞 Support

Вопросы по API: shukik85@ya.ru
