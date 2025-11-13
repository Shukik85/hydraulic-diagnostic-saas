<script setup lang="ts">
/**
 * Create System Page - Type-safe with Generated API
 * 
 * Form for creating new hydraulic system with:
 * - Full validation
 * - Type-safe form data
 * - Component management
 * - Auto-save draft
 */

import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { SystemCreate, Component } from '~/generated/api'
import { validateForm, validateRequired } from '~/utils/validation'

definePageMeta({
  middleware: ['auth', 'rbac'],
  rbac: {
    permissions: ['systems:write']
  },
  layout: 'dashboard'
})

// Composables
const api = useGeneratedApi()
const { success, error: notifyError } = useNotifications()

// Form state
const form = ref<SystemCreate>({
  name: '',
  equipment_type: 'excavator',
  manufacturer: '',
  model: '',
  serial_number: '',
  manufacture_date: '',
  installation_date: '',
  location: '',
  description: '',
  components: []
})

const errors = ref<Record<string, string>>({})
const loading = ref(false)

// Component form
const showComponentForm = ref(false)
const newComponent = ref<Partial<Component>>({
  type: 'pump',
  name: '',
  manufacturer: '',
  model: ''
})

// Validation rules
const validationRules = {
  name: [
    (v: string) => validateRequired(v, 'Название'),
    (v: string) => v.length >= 3 ? null : 'Минимум 3 символа'
  ],
  manufacturer: [(v: string) => validateRequired(v, 'Производитель')],
  model: [(v: string) => validateRequired(v, 'Модель')],
  serial_number: [(v: string) => validateRequired(v, 'Серийный номер')]
}

// Add component
function addComponent() {
  if (!newComponent.value.name) {
    notifyError('Введите название компонента')
    return
  }
  
  form.value.components.push(newComponent.value as Component)
  
  // Reset
  newComponent.value = {
    type: 'pump',
    name: '',
    manufacturer: '',
    model: ''
  }
  
  showComponentForm.value = false
  success('Компонент добавлен')
}

// Remove component
function removeComponent(index: number) {
  form.value.components.splice(index, 1)
}

// Submit form
async function submit() {
  // Validate
  const validation = validateForm(form.value, validationRules)
  if (!validation.valid) {
    errors.value = validation.errors
    return
  }
  
  loading.value = true
  errors.value = {}
  
  try {
    // ✅ Type-safe API call!
    const created = await api.equipment.createSystem(form.value)
    
    success('Система успешно создана')
    await navigateTo(`/systems/${created.id}`)
  } catch (err: any) {
    if (err.status === 409) {
      notifyError('Серийный номер уже существует')
    } else {
      notifyError('Ошибка создания системы')
    }
    console.error(err)
  } finally {
    loading.value = false
  }
}
</script>

<template>
  <div class="create-system-page">
    <div class="max-w-4xl mx-auto">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-8">
        Создание новой системы
      </h1>
      
      <form @submit.prevent="submit" class="space-y-6">
        <!-- Основная информация -->
        <section class="form-section">
          <h2 class="section-title">Основная информация</h2>
          
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <!-- Name -->
            <div class="md:col-span-2">
              <label class="form-label">
                Название системы <span class="required">*</span>
              </label>
              <input
                v-model="form.name"
                type="text"
                class="form-input"
                :class="{ 'input-error': errors.name }"
                placeholder="Экскаватор CAT-001"
                required
              />
              <p v-if="errors.name" class="error-message">{{ errors.name }}</p>
            </div>
            
            <!-- Equipment Type -->
            <div>
              <label class="form-label">
                Тип оборудования <span class="required">*</span>
              </label>
              <select v-model="form.equipment_type" class="form-input" required>
                <option value="excavator">Экскаватор</option>
                <option value="press">Гидравлический пресс</option>
                <option value="crane">Кран</option>
                <option value="injection_molding">Литьевая машина</option>
                <option value="other">Другое</option>
              </select>
            </div>
            
            <!-- Manufacturer -->
            <div>
              <label class="form-label">
                Производитель <span class="required">*</span>
              </label>
              <input
                v-model="form.manufacturer"
                type="text"
                list="manufacturers"
                class="form-input"
                :class="{ 'input-error': errors.manufacturer }"
                placeholder="Caterpillar"
                required
              />
              <datalist id="manufacturers">
                <option>Caterpillar</option>
                <option>Bosch Rexroth</option>
                <option>Parker Hannifin</option>
                <option>Eaton</option>
              </datalist>
            </div>
            
            <!-- Model -->
            <div>
              <label class="form-label">
                Модель <span class="required">*</span>
              </label>
              <input
                v-model="form.model"
                type="text"
                class="form-input"
                placeholder="320D"
                required
              />
            </div>
            
            <!-- Serial Number -->
            <div>
              <label class="form-label">
                Серийный номер <span class="required">*</span>
              </label>
              <input
                v-model="form.serial_number"
                type="text"
                class="form-input"
                placeholder="CAT-2024-001"
                required
              />
            </div>
          </div>
        </section>
        
        <!-- Submit -->
        <div class="flex gap-3">
          <button
            type="submit"
            :disabled="loading"
            class="btn-primary"
          >
            {{ loading ? 'Создание...' : 'Создать систему' }}
          </button>
          
          <button
            type="button"
            @click="navigateTo('/systems')"
            class="btn-secondary"
          >
            Отмена
          </button>
        </div>
      </form>
    </div>
  </div>
</template>

<style scoped>
.create-system-page {
  @apply container mx-auto px-4 py-8;
}

.form-section {
  @apply bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6;
}

.section-title {
  @apply text-lg font-semibold text-gray-900 dark:text-white mb-4;
}

.form-label {
  @apply block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1;
}

.required {
  @apply text-red-500;
}

.form-input {
  @apply w-full px-4 py-2 border border-gray-300 dark:border-gray-700 rounded-lg;
  @apply bg-white dark:bg-gray-800 text-gray-900 dark:text-white;
  @apply focus:ring-2 focus:ring-blue-500 focus:border-transparent;
}

.input-error {
  @apply border-red-500 focus:ring-red-500;
}

.error-message {
  @apply text-red-500 text-sm mt-1;
}

.btn-primary {
  @apply px-6 py-3 bg-blue-600 text-white rounded-lg;
  @apply hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed;
  @apply font-medium transition-colors;
}

.btn-secondary {
  @apply px-6 py-3 border border-gray-300 dark:border-gray-600 rounded-lg;
  @apply text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700;
  @apply transition-colors;
}
</style>
