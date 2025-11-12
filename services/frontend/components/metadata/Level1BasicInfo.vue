<!-- components/metadata/Level1BasicInfo.vue -->
<template>
  <div class="level-1 space-y-6">
    <div>
      <h2 class="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
        {{ $t('wizard.level1.title') }}
      </h2>
      <p class="text-sm text-gray-600 dark:text-gray-400">
        {{ $t('wizard.level1.description') }}
      </p>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
      <!-- Equipment Type -->
      <UFormGroup :label="$t('wizard.level1.equipmentType')" required>
        <USelect
          v-model="formData.equipment_type"
          :options="equipmentTypeOptions"
          :placeholder="$t('wizard.level1.selectType')"
          @update:model-value="onEquipmentTypeChange"
        />
        <template v-if="formData.equipment_type === 'other'" #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            {{ $t('wizard.level1.otherTypeHint') }}
          </p>
        </template>
      </UFormGroup>

      <!-- Manufacturer -->
      <UFormGroup :label="$t('wizard.level1.manufacturer')" required>
        <UInput
          v-model="formData.manufacturer"
          :placeholder="$t('wizard.level1.manufacturerPlaceholder')"
          icon="i-heroicons-building-office-2"
          :list="manufacturers"
          @input="onManufacturerChange"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            {{ $t('wizard.level1.manufacturerHint') }}
          </p>
        </template>
      </UFormGroup>

      <!-- Model -->
      <UFormGroup :label="$t('wizard.level1.model')" required>
        <UInput
          v-model="formData.model"
          :placeholder="$t('wizard.level1.modelPlaceholder')"
          icon="i-heroicons-tag"
        />
      </UFormGroup>

      <!-- Serial Number -->
      <UFormGroup :label="$t('wizard.level1.serialNumber')" required>
        <UInput
          v-model="formData.serial_number"
          :placeholder="$t('wizard.level1.serialNumberPlaceholder')"
          icon="i-heroicons-hashtag"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            {{ $t('wizard.level1.serialNumberHint') }}
          </p>
        </template>
      </UFormGroup>

      <!-- Manufacture Date -->
      <UFormGroup :label="$t('wizard.level1.manufactureDate')">
        <UInput
          v-model="formData.manufacture_date"
          type="date"
          :max="today"
          icon="i-heroicons-calendar"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            {{ $t('wizard.level1.manufactureDateHint') }}
          </p>
        </template>
      </UFormGroup>

      <!-- Equipment ID (auto-generated) -->
      <UFormGroup :label="$t('wizard.level1.systemId')">
        <UInput
          :model-value="generatedEquipmentId"
          disabled
          icon="i-heroicons-identification"
        />
        <template #hint>
          <p class="text-xs text-gray-500 dark:text-gray-400">
            {{ $t('wizard.level1.systemIdHint') }}
          </p>
        </template>
      </UFormGroup>
    </div>

    <!-- Validation Status -->
    <UAlert
      v-if="validationErrors.length > 0"
      color="red"
      icon="i-heroicons-exclamation-triangle"
      :title="$t('wizard.level1.validation.errors')"
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
      :title="$t('wizard.level1.validation.success')"
    />
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata'
import type { EquipmentType } from '~/types/metadata'

const { t } = useI18n()
const store = useMetadataStore()

const formData = reactive({
  equipment_type: store.wizardState.system.equipment_type || '' as EquipmentType,
  manufacturer: store.wizardState.system.manufacturer || '',
  model: store.wizardState.system.model || '',
  serial_number: store.wizardState.system.serial_number || '',
  manufacture_date: store.wizardState.system.manufacture_date || '',
})

const equipmentTypeOptions = computed(() => [
  { value: 'excavator_tracked', label: t('wizard.level1.types.excavator_tracked') },
  { value: 'excavator_wheeled', label: t('wizard.level1.types.excavator_wheeled') },
  { value: 'loader_wheeled', label: t('wizard.level1.types.loader_wheeled') },
  { value: 'crane_mobile', label: t('wizard.level1.types.crane_mobile') },
  { value: 'other', label: t('wizard.level1.types.other') }
])

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
    return t('wizard.level1.notGenerated')
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
    errors.push(t('wizard.level1.validation.noType'))
  }

  if (!formData.manufacturer) {
    errors.push(t('wizard.level1.validation.noManufacturer'))
  }

  if (!formData.model) {
    errors.push(t('wizard.level1.validation.noModel'))
  }

  if (!formData.serial_number) {
    errors.push(t('wizard.level1.validation.noSerialNumber'))
  } else if (formData.serial_number.length < 3) {
    errors.push(t('wizard.level1.validation.serialTooShort'))
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
    equipment_id: generatedEquipmentId.value !== t('wizard.level1.notGenerated')
      ? generatedEquipmentId.value
      : undefined
  })
}

watch(formData, () => {
  updateStore()
}, { deep: true })

watch(() => store.wizardState.system, (newSystem) => {
  if (!newSystem) return
  if (newSystem.equipment_type) formData.equipment_type = newSystem.equipment_type
  if (newSystem.manufacturer) formData.manufacturer = newSystem.manufacturer
  if (newSystem.model) formData.model = newSystem.model
  if (newSystem.serial_number) formData.serial_number = newSystem.serial_number
  if (newSystem.manufacture_date) formData.manufacture_date = newSystem.manufacture_date
}, { deep: true })
</script>
