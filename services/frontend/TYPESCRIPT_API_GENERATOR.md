# 🤖 TypeScript API Generator from OpenAPI

Автоматическая генерация type-safe TypeScript клиента из OpenAPI спецификаций.

---

## 🎯 Что это даёт?

### ✅ Преимущества:

1. **100% Type Safety** - полная типизация всех API endpoints
2. **Auto-sync** - автоматическое обновление при изменении backend
3. **Autocomplete** - IDE подсказки для всех методов
4. **Breaking changes detection** - компилятор находит несоответствия
5. **Zero manual work** - не нужно вручную писать types

### 📊 Результат:

```typescript
// ❌ ДО: любые типы, runtime ошибки
const data = await fetch('/api/systems').then(r => r.json())
data.forEach(item => console.log(item.name))  // Ошибка только в runtime!

// ✅ ПОСЛЕ: type-safe, compile-time проверка
const api = useGeneratedApi()
const systems = await api.equipment.getSystems()
systems.forEach(s => console.log(s.name))  // ✅ Типы проверены!
```

---

## 🚀 Quick Start

### 1. Установка зависимостей

```bash
cd services/frontend
npm install
```

Это установит:
- `openapi-typescript-codegen` - генератор TypeScript клиента
- `axios` - HTTP клиент
- `nodemon` - watch mode для auto-regeneration

### 2. Генерация клиента

```bash
# Одноразовая генерация
npm run generate:api

# Автоматическая регенерация при изменении specs
npm run generate:api:watch
```

### 3. Использование в коде

```typescript
// В любом компоненте или composable
import { useGeneratedApi } from '~/composables/useGeneratedApi'

const api = useGeneratedApi()

// Полная типизация!
const systems = await api.equipment.getSystems()
const diagnosis = await api.gnn.runDiagnosis({ systemId: 'sys_001' })
```

---

## 📁 Структура сгенерированных файлов

```
services/frontend/generated/api/
├── index.ts                 # Main export
├── core/
│   ├── ApiError.ts         # Error handling
│   ├── ApiRequestOptions.ts
│   ├── ApiResult.ts
│   ├── CancelablePromise.ts
│   └── OpenAPI.ts          # Configuration
├── models/
│   ├── System.ts           # System type
│   ├── Component.ts        # Component type
│   ├── Sensor.ts           # Sensor type
│   ├── Diagnosis.ts        # Diagnosis type
│   └── ...                 # All other models
└── services/
    ├── EquipmentService.ts # Equipment API
    ├── DiagnosisService.ts # Diagnosis API
    ├── GnnService.ts       # GNN API
    └── RagService.ts       # RAG API
```

---

## 🎓 Примеры использования

### Пример 1: Получение списка систем

```vue
<script setup lang="ts">
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System } from '~/generated/api'

const api = useGeneratedApi()
const systems = ref<System[]>([])
const loading = ref(false)

async function loadSystems() {
  loading.value = true
  try {
    systems.value = await api.equipment.getSystems()
  } catch (error) {
    console.error('Failed to load systems:', error)
  } finally {
    loading.value = false
  }
}

onMounted(() => loadSystems())
</script>

<template>
  <div>
    <div v-if="loading">Загрузка...</div>
    <div v-else>
      <div v-for="system in systems" :key="system.id">
        {{ system.name }} - {{ system.status }}
      </div>
    </div>
  </div>
</template>
```

### Пример 2: Создание новой системы

```typescript
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { SystemCreate } from '~/generated/api'

const api = useGeneratedApi()

const newSystem: SystemCreate = {
  name: 'Excavator CAT-320D',
  equipment_type: 'excavator',
  manufacturer: 'Caterpillar',
  model: '320D',
  serial_number: 'CAT-2024-001'
}

try {
  const created = await api.equipment.createSystem(newSystem)
  console.log('Created:', created.id)
} catch (error) {
  if (error.status === 409) {
    console.error('System already exists')
  }
}
```

### Пример 3: Запуск диагностики с GNN

```typescript
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { DiagnosisRequest, DiagnosisResult } from '~/generated/api'

const api = useGeneratedApi()

const request: DiagnosisRequest = {
  system_id: 'sys_001',
  sensor_readings: [
    {
      sensor_id: 'sensor_001',
      timestamp: new Date().toISOString(),
      value: 120.5,
      unit: 'bar'
    }
  ]
}

const result = await api.gnn.runDiagnosis(request)

console.log('Anomaly score:', result.anomaly_score)
console.log('Anomalies:', result.anomalies)
```

### Пример 4: RAG интерпретация

```typescript
import { useGeneratedApi } from '~/composables/useGeneratedApi'

const api = useGeneratedApi()

// После получения GNN результатов
const gnnResult = await api.gnn.runDiagnosis({ ... })

// Получить human-readable интерпретацию
const interpretation = await api.rag.interpretDiagnosis({
  gnnResult: gnnResult,
  equipmentContext: {
    equipment_type: 'excavator',
    manufacturer: 'Caterpillar'
  }
})

console.log('Summary:', interpretation.summary)
console.log('Recommendations:', interpretation.recommendations)
```

---

## ⚙️ Конфигурация

### OpenAPI Client Configuration

Настройки в `composables/useGeneratedApi.ts`:

```typescript
const apiConfig: Partial<OpenAPIConfig> = {
  BASE: 'http://localhost:8100',     // Base URL для API
  VERSION: '1.0.0',                  // API version
  WITH_CREDENTIALS: false,           // CORS credentials
  TOKEN: authStore.token,            // JWT token для аутентификации
  HEADERS: {
    'Content-Type': 'application/json'
  }
}
```

### Runtime Configuration

В `nuxt.config.ts`:

```typescript
export default defineNuxtConfig({
  runtimeConfig: {
    public: {
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8100'
    }
  }
})
```

---

## 🔄 Workflow интеграции

### Автоматическая синхронизация:

1. **Backend изменён** → OpenAPI spec обновлён
2. **CI/CD workflow** → Генерирует TypeScript клиент
3. **Auto-commit** → Коммитит изменения в frontend
4. **TypeScript compilation** → Проверяет breaking changes
5. **Tests** → Запускаются автоматически

### Ручная синхронизация:

```bash
# 1. Обновить OpenAPI specs (из backend)
cd ../..
./scripts/generate-openapi.sh

# 2. Сгенерировать TypeScript клиент
cd services/frontend
npm run generate:api

# 3. Проверить TypeScript
npm run typecheck

# 4. Запустить тесты
npm test
```

---

## 🧪 Testing

### Unit тесты с мокированием:

```typescript
import { describe, it, expect, vi } from 'vitest'
import { useGeneratedApi } from '~/composables/useGeneratedApi'

describe('useGeneratedApi', () => {
  it('should fetch systems', async () => {
    const api = useGeneratedApi()
    
    // Mock API response
    vi.spyOn(api.equipment, 'getSystems').mockResolvedValue([
      {
        id: 'sys_001',
        name: 'Test System',
        status: 'online'
      }
    ])
    
    const systems = await api.equipment.getSystems()
    expect(systems).toHaveLength(1)
    expect(systems[0].name).toBe('Test System')
  })
})
```

---

## 🐛 Troubleshooting

### Проблема: "Specs directory not found"

**Решение:**
```bash
cd ../..
./scripts/generate-openapi.sh
```

### Проблема: "TypeScript compilation errors"

**Причина:** Backend изменил API, breaking change  
**Решение:** Обновите код frontend согласно новым типам

### Проблема: "Module '~/generated/api' not found"

**Решение:**
```bash
npm run generate:api
```

### Проблема: "401 Unauthorized"

**Причина:** Отсутствует или истёк auth token  
**Решение:** Проверьте, что `authStore.token` установлен

---

## 📚 Дополнительные ресурсы

- [OpenAPI TypeScript Codegen](https://github.com/ferdikoomen/openapi-typescript-codegen)
- [OpenAPI Specification](https://swagger.io/specification/)
- [TypeScript Best Practices](https://www.typescriptlang.org/docs/handbook/2/everyday-types.html)

---

## 🎯 Best Practices

### 1. Всегда используйте типы из generated/api

```typescript
// ❌ Плохо
interface System {
  id: string
  name: string
}

// ✅ Хорошо
import type { System } from '~/generated/api'
```

### 2. Обрабатывайте ошибки

```typescript
import { handleApiError } from '~/composables/useGeneratedApi'

try {
  await api.equipment.createSystem(data)
} catch (error) {
  const message = handleApiError(error)
  notifications.error(message)
}
```

### 3. Используйте watch mode при разработке

```bash
# Terminal 1: Backend
npm run dev

# Terminal 2: Specs watch
cd services/frontend
npm run generate:api:watch

# Terminal 3: Frontend
npm run dev
```

### 4. Коммитьте сгенерированный код

**НЕ добавляйте** `generated/api/` в `.gitignore`!

Это позволит:
- ✅ Видеть breaking changes в PR diff
- ✅ Работать без доступа к backend
- ✅ Быстрее собирать проект (не нужна регенерация)

---

## 🚀 Следующие шаги

После настройки TypeScript генератора:

1. ✅ **Создать новые pages** - используя type-safe API
2. ✅ **Обновить существующие composables** - заменить ручные типы
3. ✅ **Добавить E2E тесты** - с реальными API calls
4. ✅ **Настроить CI/CD** - автоматическая проверка типов

---

**🎉 Готово! Теперь у вас type-safe API client!**
