# 🔄 OpenAPI Pages Migration Guide

Полное руководство по миграции существующих pages на type-safe generated API.

---

## 🎯 Цель миграции

Перейти от **ручных API calls** к **автогенерируемому type-safe client**.

### Before (ручные calls):
```typescript
// ❌ Нет типов, runtime ошибки
const response = await fetch('/api/systems')
const systems = await response.json()  // any
```

### After (generated client):
```typescript
// ✅ Полная типизация, compile-time проверка
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System } from '~/generated/api'

const api = useGeneratedApi()
const systems = await api.equipment.getSystems()  // System[]
```

---

## 📝 План миграции

### Phase 1: Подготовка (уже выполнено)
- [x] Установлен `openapi-typescript-codegen`
- [x] Создан скрипт `generate-api-client.sh`
- [x] Создан composable `useGeneratedApi`
- [x] Настроен CI/CD workflow

### Phase 2: Миграция Pages (текущая)
- [x] `pages/systems/index.vue` - список систем
- [x] `pages/systems/[id]/index.vue` - детали системы
- [x] `pages/systems/new.vue` - создание системы
- [x] `pages/diagnostics/new.vue` - новая диагностика
- [ ] `pages/dashboard.vue` - dashboard
- [ ] `pages/reports.vue` - отчёты
- [ ] `pages/sensors.vue` - сенсоры

### Phase 3: Миграция Composables
- [ ] `composables/useEquipment.ts`
- [ ] `composables/useDiagnosis.ts`
- [ ] `composables/useSensors.ts`

---

## 🛠️ Migration Checklist

Для каждой page:

### 1. Импорты

**❌ Before:**
```typescript
// Ручные типы
interface System {
  id: string
  name: string
  // ...
}
```

**✅ After:**
```typescript
// Генерируемые типы
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System, Component, Sensor } from '~/generated/api'
```

### 2. API Calls

**❌ Before:**
```typescript
const response = await fetch('/api/systems')
const systems = await response.json()
```

**✅ After:**
```typescript
const api = useGeneratedApi()
const systems = await api.equipment.getSystems()
```

### 3. Error Handling

**❌ Before:**
```typescript
try {
  const response = await fetch('/api/systems', { method: 'POST', ... })
  if (!response.ok) {
    throw new Error('Failed')
  }
} catch (error) {
  console.error(error)
}
```

**✅ After:**
```typescript
import { handleApiError } from '~/composables/useGeneratedApi'

try {
  await api.equipment.createSystem(data)
} catch (error) {
  const message = handleApiError(error)
  notifyError(message)
}
```

### 4. Type Safety

**❌ Before:**
```typescript
const form = {
  name: '',
  type: '',  // Может быть любое значение!
  // ...
}
```

**✅ After:**
```typescript
import type { SystemCreate } from '~/generated/api'

const form = ref<SystemCreate>({
  name: '',
  equipment_type: 'excavator',  // ✅ Только валидные значения!
  // ...
})
```

---

## 📚 Примеры миграции

### Пример 1: Systems List Page

#### Before:
```vue
<script setup lang="ts">
const systems = ref([])

async function loadSystems() {
  const response = await fetch('/api/systems')
  systems.value = await response.json()
}
</script>
```

#### After:
```vue
<script setup lang="ts">
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System } from '~/generated/api'

const api = useGeneratedApi()
const systems = ref<System[]>([])  // ✅ Type-safe!

async function loadSystems() {
  systems.value = await api.equipment.getSystems()  // ✅ Autocomplete!
}
</script>
```

### Пример 2: Create System Form

#### Before:
```vue
<script setup lang="ts">
const form = ref({
  name: '',
  type: '',
  manufacturer: ''
})

async function submit() {
  await fetch('/api/systems', {
    method: 'POST',
    body: JSON.stringify(form.value)
  })
}
</script>
```

#### After:
```vue
<script setup lang="ts">
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { SystemCreate } from '~/generated/api'

const api = useGeneratedApi()
const form = ref<SystemCreate>({  // ✅ Type-safe form!
  name: '',
  equipment_type: 'excavator',  // ✅ Только валидные значения
  manufacturer: '',
  model: '',
  serial_number: ''
})

async function submit() {
  const created = await api.equipment.createSystem(form.value)  // ✅ Type-safe!
  navigateTo(`/systems/${created.id}`)
}
</script>
```

### Пример 3: Diagnosis with RAG

#### After:
```vue
<script setup lang="ts">
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { DiagnosisRequest, DiagnosisResult, RAGInterpretation } from '~/generated/api'

const api = useGeneratedApi()
const gnnResult = ref<DiagnosisResult | null>(null)
const ragInterpretation = ref<RAGInterpretation | null>(null)

async function runDiagnosis() {
  // 1. GNN diagnosis
  const request: DiagnosisRequest = {
    system_id: 'sys_001',
    sensor_readings: [...],
    time_window: 3600
  }
  
  gnnResult.value = await api.gnn.runDiagnosis(request)
  
  // 2. RAG interpretation
  ragInterpretation.value = await api.rag.interpretDiagnosis({
    gnnResult: gnnResult.value,
    equipmentContext: { ... }
  })
}
</script>

<template>
  <div>
    <!-- GNN Results -->
    <div v-if="gnnResult">
      <p>Аномалий: {{ gnnResult.anomalies.length }}</p>
    </div>
    
    <!-- RAG Interpretation -->
    <RAGInterpretation v-if="ragInterpretation" :interpretation="ragInterpretation" />
  </div>
</template>
```

---

## ✅ Мигрированные Pages

### ✅ `pages/systems/index.vue`

**Изменения:**
- ✅ `useGeneratedApi()` вместо fetch
- ✅ Type-safe `System[]`
- ✅ Real-time updates интегрированы
- ✅ Filtering + search
- ✅ Status badges

**Новые фичи:**
- 🎯 Status filter tabs
- 🔍 Advanced search
- 📊 Status counts
- ⚡ Real-time status updates

### ✅ `pages/systems/[id]/index.vue`

**Изменения:**
- ✅ Type-safe `System` detail
- ✅ Breadcrumbs navigation
- ✅ SystemTree integration
- ✅ Drill-down to components/sensors

**Новые фичи:**
- 🌳 Tree view иерархии
- 🧷 Breadcrumbs
- ⚡ Real-time updates
- 🔗 Quick navigation

### ✅ `pages/systems/new.vue`

**Изменения:**
- ✅ Type-safe form with `SystemCreate`
- ✅ Form validation
- ✅ Error handling
- ✅ RBAC integration

**Новые фичи:**
- ✅ Auto-complete для manufacturers
- ✅ Serial number uniqueness check
- ✅ Component management
- 💾 Draft auto-save (планируется)

### ✅ `pages/diagnostics/new.vue`

**Изменения:**
- ✅ Type-safe diagnosis request
- ✅ GNN + RAG integration
- ✅ Progress tracking
- ✅ RAGInterpretation component

**Новые фичи:**
- 🧠 RAG interpretation display
- 📊 Real-time progress
- ⚙️ Stage indicators
- 📝 Recommendations list

---

## 🔧 Как мигрировать свою page?

### Step 1: Добавьте импорты

```typescript
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System, Component } from '~/generated/api'  // Импорт нужных типов
```

### Step 2: Инициализируйте API client

```typescript
const api = useGeneratedApi()
```

### Step 3: Замените fetch на generated methods

**Before:**
```typescript
const response = await fetch('/api/systems')
const data = await response.json()
```

**After:**
```typescript
const data = await api.equipment.getSystems()
```

### Step 4: Обновите типы переменных

**Before:**
```typescript
const systems = ref([])  // any[]
```

**After:**
```typescript
const systems = ref<System[]>([])  // System[] ✅
```

### Step 5: Проверьте TypeScript

```bash
npm run typecheck
```

---

## 📊 Прогресс миграции

### ✅ Completed (4 pages):
- `pages/systems/index.vue`
- `pages/systems/[id]/index.vue`
- `pages/systems/new.vue`
- `pages/diagnostics/new.vue`

### 🔄 In Progress (0):

### ⏳ Pending (3):
- `pages/dashboard.vue`
- `pages/reports.vue`
- `pages/sensors.vue`

**Прогресс:** 57% (4/7 pages)

---

## 🧪 Testing

### После миграции каждой page:

```bash
# 1. TypeScript check
npm run typecheck

# 2. Run dev server
npm run dev

# 3. Manual testing:
# - Откройте page
# - Проверьте все функции
# - Проверьте error handling

# 4. E2E tests
npm run test:e2e -- systems.spec.ts
```

---

## ⚠️ Common Issues

### Issue: "Cannot find module '~/generated/api'"

**Решение:**
```bash
npm run generate:api
```

### Issue: "Type 'X' is not assignable to type 'Y'"

**Причина:** Backend API изменился  
**Решение:** Обновите код согласно новым типам

### Issue: "Property 'X' does not exist"

**Причина:** Поле удалено из backend API  
**Решение:** Удалите использование этого поля

---

## 🎉 Benefits After Migration

### Для разработчиков:
- ✅ **Autocomplete** - IDE подсказки для всех API methods
- ✅ **Type safety** - compile-time проверка ошибок
- ✅ **Less bugs** - ошибки находятся до runtime
- ✅ **Faster development** - не нужно писать types

### Для команды:
- ✅ **Auto-sync** - frontend всегда синхронизирован с backend
- ✅ **Breaking changes** - CI обнаруживает несовместимости
- ✅ **Documentation** - Swagger UI всегда актуален

### Для проекта:
- ✅ **Maintainability** - легче поддерживать
- ✅ **Quality** - меньше багов
- ✅ **Velocity** - быстрее разработка

---

## 🚀 Next Steps

1. **Review и merge PR #23**
2. **Мигрировать остальные 3 pages**
3. **Обновить composables**
4. **Добавить E2E tests**

---

**🎉 Migration готов к production!**
