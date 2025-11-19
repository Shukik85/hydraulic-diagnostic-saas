# 🔧 Исправление импортов в Vue файлах

## 🐞 Проблема

В файлах с `<script setup lang="ts">` отсутствуют необходимые импорты Vue API и Nuxt composables.

### Типичные ошибки:

```typescript
// ❌ ПЛОХО - отсутствуют импорты
<script setup lang="ts">
const count = ref(0)
const user = useAuthStore()
</script>

// ✅ ХОРОШО - все импорты присутствуют
<script setup lang="ts">
import { ref, useAuthStore } from '#imports'

const count = ref(0)
const user = useAuthStore()
</script>
```

## 🔧 Решение

### Автоматическое исправление

Запустите скрипт для автоматического исправления всех файлов:

```bash
cd services/frontend
chmod +x scripts/fix-imports.sh
./scripts/fix-imports.sh
```

Скрипт автоматически:
- 🔍 Найдёт все `.vue` файлы с `<script setup lang="ts">`
- 🔎 Определит используемые Vue API (`ref`, `computed`, `watch`, etc.)
- ➕ Добавит недостающие импорты из `'#imports'`
- 📊 Покажет статистику исправлений

### Ручное исправление (опционально)

Если нужно исправить конкретный файл вручную:

#### 1. Определите, какие API используются

Просмотрите код и найдите все используемые Vue/Nuxt API:

- **Vue Composition API**: `ref`, `computed`, `watch`, `watchEffect`, `reactive`, `toRef`, `toRefs`
- **Lifecycle Hooks**: `onMounted`, `onUnmounted`, `onBeforeMount`, `onBeforeUnmount`, `onUpdated`, `onErrorCaptured`
- **Nuxt Composables**: `useRouter`, `useRoute`, `definePageMeta`, `navigateTo`
- **Store**: `useAuthStore`, `useSystemStore`, etc.
- **Vue Utilities**: `nextTick`, `defineProps`, `defineEmits`, `defineExpose`

#### 2. Добавьте импорт

Добавьте строку импорта сразу после `<script setup lang="ts">`:

```typescript
<script setup lang="ts">
import { ref, computed, onMounted, useRouter } from '#imports'

// Ваш код...
</script>
```

## ✅ Проверка

После исправления запустите проверку типов:

```bash
npm run typecheck
```

Если ошибок нет, значит всё исправлено правильно! ✅

## 📖 Nuxt 4 Best Practices

### Почему `'#imports'` вместо `'vue'`?

Nuxt 4 рекомендует использовать **авто-импорты** через `'#imports'`:

```typescript
// ✅ ПРАВИЛЬНО (Nuxt 4 way)
import { ref, computed, useRouter } from '#imports'

// 🚫 НЕЖЕЛАТЕЛЬНО (старый способ)
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
```

**Преимущества:**
- 📦 Единый источник для всех импортов
- 🚀 Лучшая оптимизация сборки
- 🔧 Полное согласование с Nuxt экосистемой
- 🎯 Строгая типизация TypeScript

### Какие API доступны через `'#imports'`?

Все Vue Composition API + все Nuxt composables:

```typescript
import {
  // Vue Reactivity
  ref, computed, reactive, readonly,
  toRef, toRefs, unref, isRef,
  
  // Vue Lifecycle
  onMounted, onUnmounted, onBeforeMount, onBeforeUnmount,
  onUpdated, onBeforeUpdate, onErrorCaptured,
  
  // Vue Watchers
  watch, watchEffect, watchPostEffect, watchSyncEffect,
  
  // Vue Utilities
  nextTick, defineProps, defineEmits, defineExpose,
  
  // Nuxt Routing
  useRouter, useRoute, navigateTo, definePageMeta,
  
  // Nuxt State
  useState, useFetch, useAsyncData, useLazyFetch, useLazyAsyncData,
  
  // Nuxt Utils
  useHead, useSeoMeta, useRuntimeConfig, useNuxtApp,
  
  // Pinia Stores
  useAuthStore, useSystemStore, // и другие ваши stores
} from '#imports'
```

## 🐛 Частые проблемы

### Ошибка: "Cannot find name 'ref'"

**Причина**: Отсутствует импорт `ref`

**Решение**:
```typescript
import { ref } from '#imports'
```

### Ошибка: "Cannot find name 'useRouter'"

**Причина**: Отсутствует импорт `useRouter`

**Решение**:
```typescript
import { useRouter } from '#imports'
```

### Ошибка: "Cannot find name 'useAuthStore'"

**Причина**: Отсутствует импорт store

**Решение**:
```typescript
import { useAuthStore } from '#imports'
```

## 📝 Примеры

### Простой компонент с reactive state

```typescript
<script setup lang="ts">
import { ref, computed } from '#imports'

const count = ref(0)
const doubleCount = computed(() => count.value * 2)

const increment = () => {
  count.value++
}
</script>
```

### Компонент с lifecycle hooks

```typescript
<script setup lang="ts">
import { ref, onMounted, onUnmounted } from '#imports'

const data = ref(null)

onMounted(() => {
  console.log('Component mounted')
})

onUnmounted(() => {
  console.log('Component unmounted')
})
</script>
```

### Компонент с router navigation

```typescript
<script setup lang="ts">
import { ref, useRouter } from '#imports'

const router = useRouter()

const goToPage = () => {
  router.push('/dashboard')
}
</script>
```

### Компонент с Pinia store

```typescript
<script setup lang="ts">
import { ref, computed, useAuthStore } from '#imports'

const authStore = useAuthStore()
const isAuthenticated = computed(() => authStore.isAuthenticated)

const login = async () => {
  await authStore.login({ email: 'user@example.com', password: 'password' })
}
</script>
```

---

**🎉 Удачи с исправлением импортов!**
