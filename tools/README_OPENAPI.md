# OpenAPI Tools

Автоматизация для управления OpenAPI спецификациями и генерации клиентов.

## 📦 Структура

```
tools/
├── aggregate_openapi.py          # Агрегация спек из всех сервисов
├── generate_openapi.sh           # Автоматический сбор спек
├── generate_typescript_clients.sh # Генерация TS клиентов
└── README_OPENAPI.md             # Эта документация

docs/openapi/
├── aggregated.yaml               # Полная спецификация всех API
├── backend_fastapi.json          # FastAPI Core API
├── gnn_service.json              # GNN ML Service
└── rag_service.json              # RAG Service

services/frontend/composables/api/generated/
├── backend_fastapi/              # TypeScript client для FastAPI
├── gnn_service/                  # TypeScript client для GNN
└── rag_service/                  # TypeScript client для RAG
```

## 🚀 Использование

### 1. Генерация OpenAPI спецификаций

```bash
# Запустить сервисы
docker-compose up -d backend_fastapi gnn_service rag_service

# Сгенерировать спецификации
bash tools/generate_openapi.sh

# Результат: docs/openapi/*.json
```

### 2. Агрегация спецификаций

```bash
# Объединить все спецификации в одну
python tools/aggregate_openapi.py

# Результат: docs/openapi/aggregated.yaml
```

### 3. Генерация TypeScript клиентов

```bash
# Установить openapi-generator (один раз)
npm install -g @openapitools/openapi-generator-cli

# Сгенерировать клиенты
bash tools/generate_typescript_clients.sh

# Результат: services/frontend/composables/api/generated/
```

## 📝 Использование в Frontend

### Nuxt Composables

```typescript
// composables/useApi.ts
import { DefaultApi, Configuration } from '~/composables/api/generated/backend_fastapi'

export const useApi = () => {
  const config = useRuntimeConfig()

  const apiConfig = new Configuration({
    basePath: config.public.apiUrl,
    headers: {
      'X-API-Key': useAuth().apiKey.value
    }
  })

  const api = new DefaultApi(apiConfig)

  return { api }
}
```

### Использование в компонентах

```vue
<script setup lang="ts">
const { api } = useGeneratedApi()

const { data: equipment } = await useAsyncData('equipment', () =>
  api.getEquipmentMetadata({ userId: 'user123', systemId: 'press_01' })
)
</script>
```

## 🔄 CI/CD Автоматизация

GitHub Actions автоматически обновляет OpenAPI спецификации и TypeScript клиенты при каждом push в `main` или `develop`.

См. `.github/workflows/openapi.yml`

## 🛠️ Troubleshooting

### Проблема: "Service not running"

**Решение:**
```bash
docker-compose up -d backend_fastapi gnn_service rag_service
docker-compose ps  # Проверить статус
```

### Проблема: "openapi-generator-cli not found"

**Решение:**
```bash
npm install -g @openapitools/openapi-generator-cli
```

### Проблема: Конфликты схем при агрегации

**Решение:** Скрипт автоматически добавляет префикс сервиса к именам схем:
- `User` → `backend_fastapi_User`
- `InferenceRequest` → `gnn_service_InferenceRequest`

## 📚 Дополнительно

- [OpenAPI Specification](https://spec.openapis.org/oas/latest.html)
- [OpenAPI Generator](https://openapi-generator.tech/)
- [FastAPI OpenAPI](https://fastapi.tiangolo.com/advanced/extending-openapi/)
