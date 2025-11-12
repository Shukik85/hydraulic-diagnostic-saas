# 🎨 Code Style Guide - Hydraulic Diagnostic Platform

**Version:** 1.0.0  
**Last Updated:** November 12, 2025  
**Status:** ✅ Active Standard

---

## 🎯 Philosophy

> **"Code should look like it was written by one professional developer"**

This guide ensures:
- ✅ **Consistency** - Same patterns everywhere
- ✅ **Readability** - Easy to understand
- ✅ **Maintainability** - Easy to update
- ✅ **Type Safety** - Catch errors early
- ✅ **Performance** - Optimized by default

---

## 📁 File Naming Conventions

### Components
```
✅ PascalCase
EquipmentCard.vue
DiagnosticChart.vue
UserProfile.vue

❌ Not allowed
equipment-card.vue
diagnostic_chart.vue
```

### Composables
```
✅ camelCase with 'use' prefix
useApi.ts
useWebSocket.ts
useEquipment.ts

❌ Not allowed
api.ts
websocket.composable.ts
```

### Stores
```
✅ camelCase with .store.ts suffix
auth.store.ts
equipment.store.ts
metadata.store.ts

❌ Not allowed
AuthStore.ts
auth_store.ts
```

### Pages
```
✅ kebab-case
dashboard.vue
equipment/[id].vue
settings/profile.vue

❌ Not allowed
Dashboard.vue
Equipment_Id.vue
```

### Types
```
✅ PascalCase interfaces/types
interface User {}
type Status = 'active' | 'inactive'

✅ UPPER_SNAKE_CASE for constants
const API_BASE_URL = '...'
const MAX_RETRIES = 3
```

---

## 📝 TypeScript Standards

### 1. Always Use Explicit Types

```typescript
// ✅ GOOD: Explicit types
const equipment = ref<Equipment[]>([])
const loading = ref<boolean>(false)
const count = ref<number>(0)

function processData(data: Equipment): string {
  return data.name
}

// ❌ BAD: Implicit types
const equipment = ref([])
const loading = ref(false)

function processData(data: any) {
  return data.name
}
```

### 2. No `any` Type

```typescript
// ✅ GOOD: Proper typing
function handleError(error: Error | ApiErrorResponse): string {
  if (error instanceof Error) {
    return error.message
  }
  return error.error.message
}

// ❌ BAD: Using any
function handleError(error: any) {
  return error.message || error.error?.message
}
```

### 3. Use Enums for Fixed Sets

```typescript
// ✅ GOOD: Type-safe enum
export enum EquipmentStatus {
  Active = 'active',
  Maintenance = 'maintenance',
  Inactive = 'inactive'
}

const status: EquipmentStatus = EquipmentStatus.Active

// ❌ BAD: String literals everywhere
const status = 'active'
```

### 4. Interface vs Type

```typescript
// ✅ Use interface for objects
interface Equipment {
  id: string
  name: string
}

// ✅ Use type for unions, primitives
type Status = 'active' | 'inactive'
type ID = string | number
```

---

## 🧩 Vue Component Structure

### Standard Component Template

```vue
<script setup lang="ts">
/**
 * ComponentName - Brief description
 * 
 * Features:
 * - Feature 1
 * - Feature 2
 * 
 * @example
 * <ComponentName :prop="value" @event="handler" />
 */

// 1. Imports (sorted)
import { ref, computed, watch } from 'vue'
import type { Equipment } from '~/types'

// 2. Props
interface Props {
  equipment: Equipment
  editable?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  editable: true
})

// 3. Emits
interface Emits {
  update: [equipment: Equipment]
  delete: [id: string]
}

const emit = defineEmits<Emits>()

// 4. Composables
const api = useApi()
const toast = useToast()
const { t } = useI18n()

// 5. State
const loading = ref(false)
const localData = ref({ ...props.equipment })

// 6. Computed
const isModified = computed(() => {
  return JSON.stringify(localData.value) !== JSON.stringify(props.equipment)
})

const statusColor = computed(() => {
  return props.equipment.status === 'active' ? 'green' : 'red'
})

// 7. Watchers
watch(() => props.equipment, (newVal) => {
  localData.value = { ...newVal }
}, { deep: true })

// 8. Methods (alphabetically sorted)
async function handleDelete() {
  try {
    loading.value = true
    await api.delete(`/equipment/${props.equipment.id}`)
    emit('delete', props.equipment.id)
    toast.add({ title: t('equipment.deleteSuccess'), color: 'green' })
  } catch (error) {
    toast.add({ title: t('equipment.deleteError'), color: 'red' })
  } finally {
    loading.value = false
  }
}

async function handleUpdate() {
  try {
    loading.value = true
    const response = await api.put(`/equipment/${props.equipment.id}`, localData.value)
    
    if (isApiSuccess(response)) {
      emit('update', response.data)
      toast.add({ title: t('equipment.updateSuccess'), color: 'green' })
    }
  } catch (error) {
    toast.add({ title: t('equipment.updateError'), color: 'red' })
  } finally {
    loading.value = false
  }
}

// 9. Lifecycle hooks
onMounted(() => {
  console.log('Component mounted')
})

onUnmounted(() => {
  console.log('Component cleanup')
})
</script>

<template>
  <UCard class="p-6">
    <!-- 1. Header -->
    <template #header>
      <div class="flex items-center justify-between">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-gray-100">
          {{ equipment.name }}
        </h2>
        <UBadge :color="statusColor">
          {{ equipment.status }}
        </UBadge>
      </div>
    </template>

    <!-- 2. Content -->
    <div class="space-y-4">
      <!-- Clear, semantic structure -->
    </div>

    <!-- 3. Footer -->
    <template #footer>
      <div class="flex justify-end gap-3">
        <UButton
          color="red"
          variant="outline"
          :loading="loading"
          @click="handleDelete"
        >
          {{ t('ui.delete') }}
        </UButton>
        
        <UButton
          color="primary"
          :loading="loading"
          :disabled="!isModified"
          @click="handleUpdate"
        >
          {{ t('ui.save') }}
        </UButton>
      </div>
    </template>
  </UCard>
</template>

<style scoped>
/* Only if absolutely necessary */
/* Prefer Tailwind classes */
</style>
```

---

## 🎨 Tailwind CSS Standards

### 1. Consistent Spacing

```vue
<!-- ✅ GOOD: Consistent spacing -->
<div class="space-y-6 p-6">
  <div class="flex items-center gap-3">
    <div class="p-4 rounded-lg">

<!-- ❌ BAD: Inconsistent spacing -->
<div class="space-y-2 p-3">
  <div class="flex items-center gap-2">
    <div class="p-5 rounded">
```

**Standard spacing scale:**
- Small gaps: `gap-2`, `space-y-2`
- Medium gaps: `gap-3`, `space-y-3`
- Large gaps: `gap-6`, `space-y-6`
- Padding: `p-4`, `p-6`, `p-8`

### 2. Dark Mode Always

```vue
<!-- ✅ GOOD: Dark mode for all colors -->
<div class="bg-white dark:bg-gray-800">
  <h2 class="text-gray-900 dark:text-gray-100">
  <p class="text-gray-600 dark:text-gray-400">

<!-- ❌ BAD: Missing dark mode -->
<div class="bg-white">
  <h2 class="text-gray-900">
```

### 3. Semantic Class Order

```vue
<!-- ✅ GOOD: Layout → Spacing → Visual → Text -->
<div class="flex items-center justify-between gap-3 p-6 bg-white dark:bg-gray-800 rounded-lg shadow-sm text-gray-900 dark:text-gray-100">

<!-- Order:
  1. Layout: flex, grid, block
  2. Alignment: items-center, justify-between
  3. Spacing: gap-3, p-6, space-y-4
  4. Visual: bg-*, border-*, rounded-*, shadow-*
  5. Text: text-*, font-*
-->
```

---

## 📦 Import Order

```typescript
// 1. Vue/Nuxt core
import { ref, computed, watch } from 'vue'
import { defineStore } from 'pinia'

// 2. External libraries (alphabetically)
import { z } from 'zod'
import axios from 'axios'

// 3. Internal composables
import { useApi } from '~/composables/useApi'
import { useToast } from '#imports'

// 4. Internal components (if needed)
import EquipmentCard from '~/components/equipment/EquipmentCard.vue'

// 5. Types (always after regular imports)
import type { Equipment, User } from '~/types'
import type { ApiResponse } from '~/types/api'

// 6. Constants
import { API_ENDPOINTS } from '~/constants'
```

---

## 🔤 Naming Conventions

### Variables & Functions

```typescript
// ✅ GOOD: camelCase, descriptive
const equipmentList = ref<Equipment[]>([])
const isLoading = ref(false)
const hasPermission = computed(() => true)

async function fetchEquipmentData() {}
function calculateHealthScore(data: number[]): number {}

// ❌ BAD: unclear, abbreviated
const eqList = ref([])
const loading = ref(false) // too generic
const perm = computed(() => true)

function fetch() {} // too generic
function calc(d: any) {} // abbreviated
```

### Constants

```typescript
// ✅ GOOD: UPPER_SNAKE_CASE
const API_BASE_URL = 'https://api.example.com'
const MAX_RETRY_ATTEMPTS = 3
const DEFAULT_PAGE_SIZE = 20

// ❌ BAD
const apiUrl = '...'
const maxRetries = 3
```

### Boolean Variables

```typescript
// ✅ GOOD: is/has/can prefix
const isLoading = ref(false)
const hasError = ref(false)
const canEdit = computed(() => true)
const shouldRetry = ref(true)

// ❌ BAD: unclear
const loading = ref(false)
const error = ref(false)
const edit = computed(() => true)
```

---

## 💬 Comments & Documentation

### 1. All Comments in English

```typescript
// ✅ GOOD: English comments
/**
 * Fetch equipment data from API
 * @param id - Equipment unique identifier
 * @returns Promise with equipment data
 */
async function fetchEquipment(id: string): Promise<Equipment> {
  // Validate ID format
  if (!id) throw new Error('ID required')
  
  // Make API request
  return await api.get(`/equipment/${id}`)
}

// ❌ BAD: Mixed languages
/**
 * Загрузить данные equipment from API
 */
async function fetchEquipment(id: string) {
  // Проверяем ID
}
```

### 2. JSDoc for Public Functions

```typescript
// ✅ GOOD: Complete JSDoc
/**
 * Calculate system health score based on sensor readings
 * 
 * @param readings - Array of sensor readings
 * @param thresholds - Min/max thresholds
 * @returns Health score (0-100)
 * 
 * @example
 * const score = calculateHealthScore(readings, thresholds)
 * console.log(score) // 87.5
 */
export function calculateHealthScore(
  readings: SensorReading[],
  thresholds: Thresholds
): number {
  // Implementation
}
```

### 3. Inline Comments for Complex Logic

```typescript
// ✅ GOOD: Explain why, not what
// Use exponential backoff to avoid overwhelming the server
const delay = BASE_DELAY * Math.pow(2, attempt)

// Calculate health score using weighted average
// Critical sensors have 2x weight
const score = sensors.reduce((sum, s) => {
  const weight = s.critical ? 2 : 1
  return sum + s.value * weight
}, 0)

// ❌ BAD: State the obvious
// Set loading to true
loading.value = true

// Loop through equipment
equipment.forEach(e => {})
```

---

## 🔧 Error Handling Pattern

### Standard Try-Catch Pattern

```typescript
/**
 * Standard error handling pattern
 * Use everywhere for consistency
 */
async function performAction() {
  try {
    // 1. Set loading state
    loading.value = true
    
    // 2. Perform operation
    const result = await api.doSomething()
    
    // 3. Success feedback
    toast.add({ 
      title: t('action.success'),
      description: t('action.successDesc'),
      color: 'green'
    })
    
    // 4. Return result
    return result
    
  } catch (error: any) {
    // 5. Log error
    console.error('Action failed:', error)
    
    // 6. User feedback
    toast.add({ 
      title: t('action.error'),
      description: error.message || t('action.errorDesc'),
      color: 'red'
    })
    
    // 7. Re-throw if needed
    throw error
    
  } finally {
    // 8. Always reset loading
    loading.value = false
  }
}
```

---

## 🎭 Component Patterns

### 1. Props with Defaults

```typescript
// ✅ GOOD: Explicit interface with defaults
interface Props {
  equipment: Equipment
  editable?: boolean
  showActions?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  editable: true,
  showActions: true
})

// ❌ BAD: No types
const props = defineProps({
  equipment: Object,
  editable: Boolean
})
```

### 2. Emits with Types

```typescript
// ✅ GOOD: Typed emits
interface Emits {
  update: [equipment: Equipment]
  delete: [id: string]
  close: []
}

const emit = defineEmits<Emits>()

// Usage
emit('update', equipment)
emit('delete', equipment.id)

// ❌ BAD: No types
const emit = defineEmits(['update', 'delete'])
emit('update', equipment) // No type checking!
```

### 3. Composable Usage

```typescript
// ✅ GOOD: At component top level
const api = useApi()
const toast = useToast()
const { t } = useI18n()
const route = useRoute()
const router = useRouter()

// ❌ BAD: Inside functions
function doSomething() {
  const api = useApi() // Wrong!
}
```

---

## 🎨 UI Component Standards

### 1. Always Use Nuxt UI

```vue
<!-- ✅ GOOD: Nuxt UI components -->
<UButton color="primary" @click="handle">
  {{ t('ui.save') }}
</UButton>

<UCard class="p-6">
  <template #header>
    <h2>Title</h2>
  </template>
</UCard>

<UFormGroup :label="t('form.email')" required>
  <UInput v-model="email" type="email" />
</UFormGroup>

<!-- ❌ BAD: Custom components -->
<BaseButton variant="primary" @click="handle">
<button class="custom-btn">
<div class="card">
```

### 2. Loading States

```vue
<!-- ✅ GOOD: Nuxt UI loading components -->
<div v-if="loading">
  <USkeleton class="h-32 w-full" :count="3" />
</div>

<div v-else>
  <!-- Content -->
</div>

<!-- Button loading -->
<UButton :loading="isSubmitting" @click="submit">
  {{ t('ui.submit') }}
</UButton>

<!-- ❌ BAD: Just text -->
<div v-if="loading">Loading...</div>
```

### 3. Empty States

```vue
<!-- ✅ GOOD: Helpful empty state -->
<div v-if="equipment.length === 0" class="text-center py-12">
  <UIcon 
    name="i-heroicons-cube" 
    class="w-16 h-16 text-gray-400 mx-auto mb-4"
  />
  <h3 class="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-2">
    {{ t('equipment.noEquipment') }}
  </h3>
  <p class="text-gray-600 dark:text-gray-400 mb-4">
    {{ t('equipment.noEquipmentDesc') }}
  </p>
  <UButton color="primary" @click="addEquipment">
    {{ t('equipment.add') }}
  </UButton>
</div>

<!-- ❌ BAD: Just text -->
<div v-if="equipment.length === 0">
  No equipment found
</div>
```

---

## 🌍 i18n Standards

### 1. Always Use $t()

```vue
<!-- ✅ GOOD: All text via i18n -->
<template>
  <h1>{{ $t('dashboard.title') }}</h1>
  <p>{{ $t('dashboard.subtitle') }}</p>
  <UButton>{{ $t('ui.save') }}</UButton>
</template>

<!-- ❌ BAD: Hardcoded text -->
<template>
  <h1>Dashboard</h1>
  <p>Real-time monitoring</p>
  <UButton>Save</UButton>
</template>
```

### 2. Structured Translation Keys

```json
// ✅ GOOD: Hierarchical structure
{
  "equipment": {
    "list": {
      "title": "Equipment List",
      "add": "Add Equipment"
    },
    "card": {
      "status": "Status",
      "health": "Health"
    }
  }
}

// ❌ BAD: Flat structure
{
  "equipmentListTitle": "Equipment List",
  "equipmentAdd": "Add Equipment"
}
```

---

## 🚀 Performance Patterns

### 1. Lazy Loading

```typescript
// ✅ GOOD: Lazy load heavy components
const DiagnosticChart = defineAsyncComponent(
  () => import('~/components/diagnostics/DiagnosticChart.vue')
)

const MetadataWizard = defineAsyncComponent({
  loader: () => import('~/components/metadata/MetadataWizard.vue'),
  loadingComponent: () => h('div', { class: 'loading' }, 'Loading...'),
  delay: 200
})
```

### 2. Computed vs Methods

```typescript
// ✅ GOOD: Computed for derived state
const healthColor = computed(() => {
  return props.health > 80 ? 'green' : 'red'
})

// ✅ GOOD: Methods for actions
function refreshData() {
  fetchEquipment()
}

// ❌ BAD: Method for derived state (runs every render)
function getHealthColor() {
  return props.health > 80 ? 'green' : 'red'
}
```

### 3. Watch Wisely

```typescript
// ✅ GOOD: Specific watchers
watch(() => props.equipmentId, (newId) => {
  fetchEquipment(newId)
})

watch([filter, page], () => {
  loadData()
})

// ❌ BAD: Deep watch on large objects
watch(store.state, () => {
  // Runs on every state change!
}, { deep: true })
```

---

## 🔐 Security Standards

### 1. Never Expose Tokens

```typescript
// ✅ GOOD: Token in httpOnly cookie or hidden ref
const token = useCookie('auth_token', {
  httpOnly: true,
  secure: true,
  sameSite: 'strict'
})

// ❌ BAD: Token in localStorage or reactive state
const token = ref(localStorage.getItem('token'))
```

### 2. Validate All Inputs

```typescript
// ✅ GOOD: Zod validation
import { z } from 'zod'

const LoginSchema = z.object({
  email: z.string().email(),
  password: z.string().min(8)
})

function validateLogin(data: unknown) {
  return LoginSchema.parse(data)
}

// ❌ BAD: No validation
function login(data: any) {
  api.login(data.email, data.password)
}
```

---

## 📊 State Management

### 1. Pinia Store Structure

```typescript
// ✅ GOOD: Composition API style
export const useEquipmentStore = defineStore('equipment', () => {
  // State
  const items = ref<Equipment[]>([])
  const loading = ref(false)
  
  // Getters
  const activeItems = computed(() => 
    items.value.filter(e => e.status === 'active')
  )
  
  // Actions
  async function fetch() {
    loading.value = true
    try {
      const response = await api.get('/equipment')
      if (isApiSuccess(response)) {
        items.value = response.data
      }
    } finally {
      loading.value = false
    }
  }
  
  return { items, loading, activeItems, fetch }
})

// ❌ BAD: Options API style (old)
export const useEquipmentStore = defineStore('equipment', {
  state: () => ({
    items: []
  }),
  // ...
})
```

---

## ✅ Testing Patterns

### 1. Component Tests

```typescript
// test/components/EquipmentCard.spec.ts
import { mount } from '@vue/test-utils'
import EquipmentCard from '~/components/equipment/EquipmentCard.vue'

describe('EquipmentCard', () => {
  it('renders equipment name', () => {
    const wrapper = mount(EquipmentCard, {
      props: {
        equipment: {
          id: '1',
          name: 'Test Equipment',
          status: 'active'
        }
      }
    })
    
    expect(wrapper.text()).toContain('Test Equipment')
  })
})
```

---

## 🎯 Commit Message Format

```bash
# Format: <type>(<scope>): <subject>

# Types:
feat:     New feature
fix:      Bug fix
refactor: Code refactoring
docs:     Documentation
style:    Formatting (no code change)
test:     Adding tests
chore:    Build/tools updates

# Examples:
feat(equipment): add sensor management UI
fix(api): handle network timeout errors
refactor(auth): migrate to composition API
docs(style): add code style guide
```

---

## 📋 Checklist Before Commit

- [ ] TypeScript errors fixed (`npm run type-check`)
- [ ] Linter passing (`npm run lint`)
- [ ] No console.log in production code
- [ ] All text via $t()
- [ ] Dark mode classes added
- [ ] Loading states implemented
- [ ] Error handling added
- [ ] Comments in English
- [ ] Types explicit (no `any`)
- [ ] Imports sorted

---

## 🏆 Quality Standards

### Component Quality Checklist:

- [ ] TypeScript strict mode
- [ ] Props/Emits typed
- [ ] Dark mode support
- [ ] Loading skeleton
- [ ] Empty state
- [ ] Error handling
- [ ] i18n complete
- [ ] Responsive design
- [ ] Accessibility basics
- [ ] JSDoc for public methods

---

## 🚀 Resources

- [Vue 3 Style Guide](https://vuejs.org/style-guide/)
- [Nuxt 3 Best Practices](https://nuxt.com/docs/guide/going-further/experimental-features)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Tailwind CSS](https://tailwindcss.com/docs)
- [Nuxt UI](https://ui.nuxt.com/)

---

**Follow this guide for consistent, professional code!** ✅
