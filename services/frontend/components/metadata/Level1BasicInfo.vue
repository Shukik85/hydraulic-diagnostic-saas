<!-- components/metadata/Level1BasicInfo.vue -->
<template>
  <div class="level-1 space-y-6">
    <div>
      <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
        1. Базовая информация об оборудовании
      </h2>
      <p class="text-sm text-gray-600 dark:text-gray-400">
        Укажите основные характеристики вашего оборудования
      </p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <!-- Тип оборудования -->
      <UFormGroup label="Тип оборудования" required>
        <USelect
          v-model="formData.equipment_type"
          :options="equipmentTypeOptions"
          placeholder="Выберите тип"
          @update:model-value="onEquipmentTypeChange"
        />
        <template v-if="formData.equipment_type === 'other'" #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            Опишите тип оборудования в поле "Модель"
          </p>
        </template>
      </UFormGroup>

      <!-- Производитель -->
      <UFormGroup label="Производитель" required>
        <UInput
          v-model="formData.manufacturer"
          placeholder="Например: Caterpillar, Komatsu, Volvo"
          icon="i-heroicons-building-office-2"
          :list="manufacturers"
          @input="onManufacturerChange"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            Начните вводить название — появятся подсказки
          </p>
        </template>
      </UFormGroup>

      <!-- Модель -->
      <UFormGroup label="Модель" required>
        <UInput
          v-model="formData.model"
          placeholder="Например: 320D, PC200, EC210"
          icon="i-heroicons-tag"
        />
      </UFormGroup>

      <!-- Серийный номер -->
      <UFormGroup label="Серийный номер / ID" required>
        <UInput
          v-model="formData.serial_number"
          placeholder="Уникальный идентификатор"
          icon="i-heroicons-hashtag"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            Используется для идентификации оборудования в системе
          </p>
        </template>
      </UFormGroup>

      <!-- Дата выпуска -->
      <UFormGroup label="Дата выпуска">
        <UInput
          v-model="formData.manufacture_date"
          type="date"
          :max="today"
          icon="i-heroicons-calendar"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            Влияет на расчёт возраста системы и износа
          </p>
        </template>
      </UFormGroup>

      <!-- Equipment ID (автоматически генерируется) -->
      <UFormGroup label="ID системы">
        <UInput
          :model-value="generatedEquipmentId"
          disabled
          icon="i-heroicons-identification"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            Автоматически создаётся на основе введённых данных
          </p>
        </template>
      </UFormGroup>
    </div>

    <!-- Validation Status -->
    <UAlert
      v-if="validationErrors.length > 0"
      color="red"
      icon="i-heroicons-exclamation-triangle"
      title="Ошибки валидации"
    >
      <template #description>
        <ul class="list-disc pl-5 space-y-1">
          <li v-for="error in validationErrors" :key="error" class="text-sm">
            {{ error }}
          </li>
        </ul>
      </template>
    </UAlert>

    <UAlert
      v-else-if="isFormValid"
      color="green"
      icon="i-heroicons-check-circle"
      title="Базовая информация заполнена корректно"
    />
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'
import type { EquipmentType } from '~/types/metadata'

const store = useMetadataStore()

const formData = reactive({
  equipment_type: store.wizardState.system.equipment_type || '' as EquipmentType,
  manufacturer: store.wizardState.system.manufacturer || '',
  model: store.wizardState.system.model || '',
  serial_number: store.wizardState.system.serial_number || '',
  manufacture_date: store.wizardState.system.manufacture_date || '',
})

const equipmentTypeOptions = [
  { value: 'excavator_tracked', label: 'Экскаватор гусеничный' },
  { value: 'excavator_wheeled', label: 'Экскаватор колёсный' },
  { value: 'loader_wheeled', label: 'Погрузчик колёсный' },
  { value: 'crane_mobile', label: 'Кран автомобильный' },
  { value: 'other', label: 'Другое' }
]

const manufacturers = [
  'Caterpillar',
  'Komatsu',
  'Volvo',
  'Hitachi',
  'Kobelco',
  'Doosan',
  'Hyundai',
  'JCB',
  'Liebherr',
  'SANY',
  'XCMG',
  'John Deere',
  'Case',
  'Bobcat'
]

const today = computed(() => {
  return new Date().toISOString().split('T')[0]
})

const generatedEquipmentId = computed(() => {
  if (!formData.equipment_type || !formData.serial_number) {
    return 'Не сгенерирован'
  }
  const prefix = formData.equipment_type!.split('_')[0].toUpperCase().slice(0, 2)
  const year = formData.manufacture_date
    ? new Date(formData.manufacture_date).getFullYear()
    : new Date().getFullYear()
  return `${prefix}-${year}-${formData.serial_number.slice(0, 6).toUpperCase()}`
})

const validationErrors = computed(() => {
  const errors: string[] = []

  if (!formData.equipment_type) {
    errors.push('Не выбран тип оборудования')
  }

  if (!formData.manufacturer) {
    errors.push('Не указан производитель')
  }

  if (!formData.model) {
    errors.push('Не указана модель')
  }

  if (!formData.serial_number) {
    errors.push('Не указан серийный номер')
  } else if (formData.serial_number.length < 3) {
    errors.push('Серийный номер должен содержать минимум 3 символа')
  }

  return errors
})

const isFormValid = computed(() => validationErrors.value.length === 0)

function onEquipmentTypeChange() {
  updateStore()
}

function onManufacturerChange() {
  updateStore()
}

function updateStore() {
  store.updateBasicInfo({
    equipment_type: formData.equipment_type || undefined,
    manufacturer: formData.manufacturer || undefined,
    model: formData.model || undefined,
    serial_number: formData.serial_number || undefined,
    manufacture_date: formData.manufacture_date || undefined,
    equipment_id: generatedEquipmentId.value !== 'Не сгенерирован'
      ? generatedEquipmentId.value
      : undefined
  })
}

// Watch для обновления store при изменениях
watch(formData, () => {
  updateStore()
}, { deep: true })

// Если есть изменения в store (например, загружены из localStorage)
watch(() => store.wizardState.system, (newSystem) => {
  if (!newSystem) return
  if (newSystem.equipment_type) formData.equipment_type = newSystem.equipment_type
  if (newSystem.manufacturer) formData.manufacturer = newSystem.manufacturer
  if (newSystem.model) formData.model = newSystem.model
  if (newSystem.serial_number) formData.serial_number = newSystem.serial_number
  if (newSystem.manufacture_date) formData.manufacture_date = newSystem.manufacture_date
}, { deep: true })
</script>
