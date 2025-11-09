<!-- components/metadata/Level1BasicInfo.vue -->
<template>
  <div class="level-1">
    <h2 class="text-xl font-semibold mb-6">1. Базовая информация об оборудовании</h2>

    <div class="form-grid">
      <!-- Тип оборудования -->
      <div class="form-group">
        <label class="form-label required">Тип оборудования</label>
        <select v-model="formData.equipment_type" class="form-select" @change="onEquipmentTypeChange">
          <option value="">Выберите тип</option>
          <option value="excavator_tracked">Экскаватор гусеничный</option>
          <option value="excavator_wheeled">Экскаватор колёсный</option>
          <option value="loader_wheeled">Погрузчик колёсный</option>
          <option value="crane_mobile">Кран автомобильный</option>
          <option value="other">Другое</option>
        </select>
        <p v-if="formData.equipment_type === 'other'" class="help-text">
          Опишите тип оборудования в поле "Модель"
        </p>
      </div>

      <!-- Производитель -->
      <div class="form-group">
        <label class="form-label required">Производитель</label>
        <input v-model="formData.manufacturer" type="text" class="form-input"
          placeholder="Например: Caterpillar, Komatsu, Volvo" list="manufacturers" @input="onManufacturerChange" />
        <datalist id="manufacturers">
          <option v-for="mfr in manufacturers" :key="mfr" :value="mfr" />
        </datalist>
        <p class="help-text">
          Начните вводить название — появятся подсказки
        </p>
      </div>

      <!-- Модель -->
      <div class="form-group">
        <label class="form-label required">Модель</label>
        <input v-model="formData.model" type="text" class="form-input" placeholder="Например: 320D, PC200, EC210" />
      </div>

      <!-- Серийный номер -->
      <div class="form-group">
        <label class="form-label required">Серийный номер / ID</label>
        <input v-model="formData.serial_number" type="text" class="form-input" placeholder="Уникальный идентификатор" />
        <p class="help-text">
          Используется для идентификации оборудования в системе
        </p>
      </div>

      <!-- Дата выпуска -->
      <div class="form-group">
        <label class="form-label">Дата выпуска</label>
        <input v-model="formData.manufacture_date" type="date" class="form-input" :max="today" />
        <p class="help-text">
          Влияет на расчёт возраста системы и износа
        </p>
      </div>

      <!-- Equipment ID (автоматически генерируется) -->
      <div class="form-group">
        <label class="form-label">ID системы (генерируется автоматически)</label>
        <input :value="generatedEquipmentId" type="text" class="form-input" disabled />
        <p class="help-text">
          Автоматически создаётся на основе введённых данных
        </p>
      </div>
    </div>

    <!-- Validation Status -->
    <div v-if="validationErrors.length > 0" class="validation-errors">
      <h3 class="text-sm font-semibold text-red-700 mb-2">Ошибки валидации:</h3>
      <ul class="list-disc pl-5">
        <li v-for="error in validationErrors" :key="error" class="text-red-600 text-sm">
          {{ error }}
        </li>
      </ul>
    </div>

    <div v-else-if="isFormValid" class="validation-success">
      ✓ Базовая информация заполнена корректно
    </div>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata';
import type { EquipmentType } from '~/types/metadata';

const store = useMetadataStore();

const formData = reactive({
  equipment_type: store.wizardState.system.equipment_type || '' as EquipmentType,
  manufacturer: store.wizardState.system.manufacturer || '',
  model: store.wizardState.system.model || '',
  serial_number: store.wizardState.system.serial_number || '',
  manufacture_date: store.wizardState.system.manufacture_date || '',
});

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
];

const today = computed(() => {
  return new Date().toISOString().split('T')[0];
});

const generatedEquipmentId = computed(() => {
  if (!formData.equipment_type || !formData.serial_number) {
    return 'Не сгенерирован';
  }
  const prefix = formData.equipment_type!.split('_')[0].toUpperCase().slice(0, 2);
  const year = formData.manufacture_date
    ? new Date(formData.manufacture_date).getFullYear()
    : new Date().getFullYear();
  return `${prefix}-${year}-${formData.serial_number.slice(0, 6).toUpperCase()}`;
});

const validationErrors = computed(() => {
  const errors: string[] = [];

  if (!formData.equipment_type) {
    errors.push('Не выбран тип оборудования');
  }

  if (!formData.manufacturer) {
    errors.push('Не указан производитель');
  }

  if (!formData.model) {
    errors.push('Не указана модель');
  }

  if (!formData.serial_number) {
    errors.push('Не указан серийный номер');
  } else if (formData.serial_number.length < 3) {
    errors.push('Серийный номер должен содержать минимум 3 символа');
  }

  return errors;
});

const isFormValid = computed(() => validationErrors.value.length === 0);

function onEquipmentTypeChange() {
  updateStore();
}

function onManufacturerChange() {
  updateStore();
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
  });
}

// Watch для обновления store при изменениях
watch(formData, () => {
  updateStore();
}, { deep: true });

// Если есть изменения в store (например, загружены из localStorage)
watch(() => store.wizardState.system, (newSystem) => {
  if (!newSystem) return;
  if (newSystem.equipment_type) formData.equipment_type = newSystem.equipment_type;
  if (newSystem.manufacturer) formData.manufacturer = newSystem.manufacturer;
  if (newSystem.model) formData.model = newSystem.model;
  if (newSystem.serial_number) formData.serial_number = newSystem.serial_number;
  if (newSystem.manufacture_date) formData.manufacture_date = newSystem.manufacture_date;
}, { deep: true });
</script>

<style scoped>
.level-1 {
  padding: 1rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.form-label {
  font-weight: 500;
  font-size: 0.875rem;
  color: #374151;
}

.form-label.required::after {
  content: ' *';
  color: #ef4444;
}

.form-input,
.form-select {
  padding: 0.625rem 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  transition: border-color 0.2s;
}

.form-input:focus,
.form-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.form-input:disabled {
  background: #f3f4f6;
  color: #6b7280;
  cursor: not-allowed;
}

.help-text {
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 0.25rem;
}

.validation-errors {
  padding: 1rem;
  background: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.5rem;
  margin-top: 1rem;
}

.validation-success {
  padding: 1rem;
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  border-radius: 0.5rem;
  color: #065f46;
  font-weight: 500;
  margin-top: 1rem;
}
</style>
