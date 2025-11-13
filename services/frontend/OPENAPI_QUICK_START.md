# ⚡ OpenAPI TypeScript - Quick Start

5-минутное руководство по началу работы с type-safe API client.

---

## 🚀 Quick Start (3 шага)

### 1. Установите зависимости

```bash
cd services/frontend
npm install
```

### 2. Сгенерируйте API client

```bash
npm run generate:api
```

### 3. Используйте в коде

```typescript
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System } from '~/generated/api'

const api = useGeneratedApi()
const systems = await api.equipment.getSystems()  // ✅ Type-safe!
```

**Готово!** 🎉

---

## 📝 Примеры

### Пример 1: Получить список систем

```vue
<script setup lang="ts">
import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System } from '~/generated/api'

const api = useGeneratedApi()
const systems = ref<System[]>([])

async function load() {
  systems.value = await api.equipment.getSystems()
}

onMounted(() => load())
</script>

<template>
  <div v-for="s in systems" :key="s.id">
    {{ s.name }}
  </div>
</template>
```

### Пример 2: Создать систему

```typescript
import type { SystemCreate } from '~/generated/api'

const form: SystemCreate = {
  name: 'Excavator CAT-001',
  equipment_type: 'excavator',
  manufacturer: 'Caterpillar',
  model: '320D',
  serial_number: 'CAT-2024-001'
}

const created = await api.equipment.createSystem(form)
navigat eTo(`/systems/${created.id}`)
```

### Пример 3: Запустить диагностику

```typescript
import type { DiagnosisRequest } from '~/generated/api'

const request: DiagnosisRequest = {
  system_id: 'sys_001',
  sensor_readings: [...],
  time_window: 3600
}

// GNN
const gnnResult = await api.gnn.runDiagnosis(request)

// RAG
const interpretation = await api.rag.interpretDiagnosis({
  gnnResult,
  equipmentContext: { ... }
})
```

---

## ⚙️ Commands

```bash
# Генерация (одноразово)
npm run generate:api

# Watch mode (авто-регенерация)
npm run generate:api:watch

# Type check
npm run typecheck

# Dev server
npm run dev
```

---

## 📚 Документация

### Полные руководства:

1. **TYPESCRIPT_API_GENERATOR.md** - полное руководство
   - Configuration
   - 4 примера
   - Troubleshooting
   - Best practices

2. **OPENAPI_PAGES_MIGRATION.md** - migration guide
   - План миграции
   - Checklist
   - Before/After примеры

3. **docs/OPENAPI_INTEGRATION_SUMMARY.md** - overview
   - Architecture
   - Benefits
   - Metrics

---

## ✅ Что уже работает

### Мигрированные pages:
- ✅ `/systems` - список систем
- ✅ `/systems/[id]` - детали системы
- ✅ `/systems/new` - создание
- ✅ `/diagnostics/new` - диагностика

### Новые компоненты:
- ✅ `<RAGInterpretation>` - RAG results
- ✅ `<SystemTree>` - tree view
- ✅ `<SystemBreadcrumbs>` - navigation

### CI/CD:
- ✅ Auto-sync workflow
- ✅ Breaking changes detection
- ✅ PR comments

---

## 🐛 Troubleshooting

### "❌ Cannot find module '~/generated/api'"

**Fix:**
```bash
npm run generate:api
```

### "❌ TypeScript errors"

**Причина:** Backend API changed  
**Fix:** Обновите код согласно новым types

### "❌ 401 Unauthorized"

**Причина:** Нет auth token  
**Fix:** Login через `/auth/login`

---

## 🚀 Next Steps

1. Прочитайте **TYPESCRIPT_API_GENERATOR.md**
2. Посмотрите migrated pages в `pages/systems/`
3. Начните использовать `useGeneratedApi()`!

---

**❤️ Happy coding with type-safe API!**
