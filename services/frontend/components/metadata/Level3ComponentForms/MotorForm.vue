<!-- components/metadata/Level3ComponentForms/MotorForm.vue -->
<template>
  <div class="motor-form">
    <h3 class="text-lg font-semibold mb-4">Настройка гидромотора: {{ componentId }}</h3>

    <div class="form-grid">
      <!-- Тип мотора -->
      <div class="form-group col-span-2">
        <label class="form-label required">Тип гидромотора</label>
        <select v-model="formData.motor_type" class="form-select">
          <option value="">Выберите тип</option>
          <option value="axial_piston">Аксиально-поршневой</option>
          <option value="radial_piston">Радиально-поршневой</option>
          <option value="vane">Пластинчатый</option>
          <option value="gear">Шестеренчатый</option>
        </select>
      </div>

      <!-- Рабочий объём -->
      <div class="form-group">
        <label class="form-label required">Рабочий объём (см³/об)</label>
        <input v-model.number="formData.displacement" type="number" class="form-input" placeholder="10-500 см³/об"
          min="10" max="500" />
        <p class="help-text">Определяет скорость и крутящий момент</p>
      </div>

      <!-- Максимальное давление -->
      <div class="form-group">
        <label class="form-label required">Максимальное давление (бар)</label>
        <input v-model.number="formData.max_pressure" type="number" class="form-input" placeholder="80-350 бар" min="80"
          max="350" />
      </div>

      <!-- Характер нагрузки -->
      <div class="form-group col-span-2">
        <label class="form-label required">Характер нагрузки</label>
        <select v-model="formData.load_character" class="form-select" @change="onLoadCharacterChange">
          <option value="constant">Постоянная (вращение на один угол)</option>
          <option value="variable">Переменная (качающийся момент)</option>
          <option value="impact">Ударная (толчки, кратковременные всплески)</option>
          <option value="cyclic">Циклическая (повторяющиеся паттерны)</option>
        </select>
      </div>

      <!-- Условные поля для переменной/ударной нагрузки -->
      <template v-if="formData.load_character === 'variable' || formData.load_character === 'impact'">
        <div class="form-group">
          <label class="form-label">Частота скачков (раз/мин)</label>
          <input v-model.number="formData.spike_frequency" type="number" class="form-input" placeholder="0-60" min="0"
            max="60" />
          <p class="help-text">Для расчёта вибрации и резонанса</p>
        </div>

        <div class="form-group">
          <label class="form-label">Амплитуда скачка (% от P_max)</label>
          <input v-model.number="formData.spike_amplitude" type="number" class="form-input" placeholder="0-100%" min="0"
            max="100" />
          <p class="help-text">Влияет на риск кавитации</p>
        </div>
      </template>

      <!-- Нормальный диапазон давления -->
      <div class="form-group col-span-2">
        <label class="form-label">Стабильный диапазон давления (бар)</label>
        <div class="range-input">
          <input v-model.number="formData.normal_pressure_min" type="number" class="form-input" placeholder="Мин" />
          <span class="range-separator">—</span>
          <input v-model.number="formData.normal_pressure_max" type="number" class="form-input" placeholder="Макс" />
        </div>
        <div class="smart-default" v-if="formData.max_pressure && !formData.normal_pressure_min">
          <button @click="applyPressureRange" class="default-btn">
            Рекомендуется: [{{ Math.round(formData.max_pressure * 0.5) }}, {{ Math.round(formData.max_pressure * 0.85)
            }}] бар
          </button>
        </div>
      </div>

      <!-- Нормальная температура -->
      <div class="form-group col-span-2">
        <label class="form-label">Нормальная температура работы (°C)</label>
        <div class="range-input">
          <input v-model.number="formData.normal_temp_min" type="number" class="form-input" placeholder="Мин" />
          <span class="range-separator">—</span>
          <input v-model.number="formData.normal_temp_max" type="number" class="form-input" placeholder="Макс" />
        </div>
        <p class="help-text text-amber-600">⚠ Критично: > 85°C для гидромоторов</p>
      </div>

      <!-- История обслуживания -->
      <div class="form-group col-span-2 section-header">
        <h4 class="text-md font-semibold">История обслуживания</h4>
      </div>

      <div class="form-group">
        <label class="form-label">Дата последнего ТО</label>
        <input v-model="formData.last_maintenance" type="date" class="form-input" :max="today" />
      </div>

      <div class="form-group">
        <label class="form-label">Периодичность ТО (часы)</label>
        <input v-model.number="formData.maintenance_interval_hours" type="number" class="form-input"
          placeholder="500-1000 ч" />
      </div>

      <div class="form-group">
        <label class="form-label">Текущая наработка (часы)</label>
        <input v-model.number="formData.operating_hours" type="number" class="form-input"
          placeholder="Примерная наработка" />
      </div>
    </div>

    <!-- Confidence indicator -->
    <div class="confidence-indicator" :class="confidenceClass">
      <div class="confidence-bar">
        <div class="confidence-fill" :style="{ width: `${overallConfidence * 100}%` }"></div>
      </div>
      <p class="confidence-text">
        Полнота данных: {{ Math.round(overallConfidence * 100) }}%
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'

import { useMetadataStore } from '~/stores/metadata';
import type { MotorSpecific } from '~/types/metadata';

const props = defineProps<{ componentId: string }>();
const store = useMetadataStore();

const component = computed(() =>
  store.wizardState.system.components?.find(c => c.id === props.componentId)
);

const formData = reactive<MotorSpecific & any>({
  motor_type: component.value?.motor_specific?.motor_type || '' as any,
  displacement: component.value?.motor_specific?.displacement || undefined,
  load_character: component.value?.motor_specific?.load_character || 'constant',
  spike_frequency: component.value?.motor_specific?.spike_frequency,
  spike_amplitude: component.value?.motor_specific?.spike_amplitude,
  max_pressure: component.value?.max_pressure,
  normal_pressure_min: component.value?.normal_ranges?.pressure?.min,
  normal_pressure_max: component.value?.normal_ranges?.pressure?.max,
  normal_temp_min: component.value?.normal_ranges?.temperature?.min,
  normal_temp_max: component.value?.normal_ranges?.temperature?.max,
  last_maintenance: component.value?.last_maintenance,
  maintenance_interval_hours: component.value?.maintenance_interval_hours,
  operating_hours: component.value?.operating_hours,
});

const today = computed(() => new Date().toISOString().split('T')[0]);

const overallConfidence = computed(() => {
  let filled = 0;
  let total = 4;

  if (formData.motor_type) filled++;
  if (formData.displacement) filled++;
  if (formData.max_pressure) filled++;
  if (formData.load_character) filled++;

  return filled / total;
});

const confidenceClass = computed(() => {
  if (overallConfidence.value < 0.5) return 'confidence-low';
  if (overallConfidence.value < 0.7) return 'confidence-medium';
  return 'confidence-high';
});

function applyPressureRange() {
  if (formData.max_pressure) {
    formData.normal_pressure_min = Math.round(formData.max_pressure * 0.5);
    formData.normal_pressure_max = Math.round(formData.max_pressure * 0.85);
  }
}

function onLoadCharacterChange() {
  if (formData.load_character === 'constant') {
    formData.spike_frequency = undefined;
    formData.spike_amplitude = undefined;
  }
  updateStore();
}

function updateStore() {
  store.updateComponent(props.componentId, {
    motor_specific: {
      motor_type: formData.motor_type,
      displacement: formData.displacement!,
      load_character: formData.load_character,
      spike_frequency: formData.spike_frequency,
      spike_amplitude: formData.spike_amplitude,
    },
    max_pressure: formData.max_pressure,
    normal_ranges: {
      pressure: formData.normal_pressure_min && formData.normal_pressure_max
        ? { min: formData.normal_pressure_min, max: formData.normal_pressure_max, unit: 'bar' }
        : undefined,
      temperature: formData.normal_temp_min && formData.normal_temp_max
        ? { min: formData.normal_temp_min, max: formData.normal_temp_max, unit: '°C' }
        : undefined,
    },
    last_maintenance: formData.last_maintenance,
    maintenance_interval_hours: formData.maintenance_interval_hours,
    operating_hours: formData.operating_hours,
    confidence_scores: { overall: overallConfidence.value }
  });
}

watch(formData, updateStore, { deep: true });
</script>

<style scoped>
.motor-form {
  padding: 1rem;
}

.form-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.col-span-2 {
  grid-column: span 2;
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
}

.form-input:focus,
.form-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.help-text {
  font-size: 0.75rem;
  color: #6b7280;
}

.smart-default {
  display: flex;
  padding: 0.5rem;
  background: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 0.375rem;
}

.default-btn {
  flex: 1;
  padding: 0.25rem 0.5rem;
  background: #3b82f6;
  color: white;
  border: none;
  border-radius: 0.25rem;
  font-size: 0.75rem;
  cursor: pointer;
}

.range-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.range-separator {
  color: #9ca3af;
  font-weight: 500;
}

.section-header {
  margin-top: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #e5e7eb;
}

.confidence-indicator {
  margin-top: 2rem;
  padding: 1rem;
  border-radius: 0.5rem;
}

.confidence-low {
  background: #fef2f2;
  border: 1px solid #fecaca;
}

.confidence-medium {
  background: #fffbeb;
  border: 1px solid #fde68a;
}

.confidence-high {
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
}

.confidence-bar {
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 0.5rem;
}

.confidence-fill {
  height: 100%;
  background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
  transition: width 0.3s;
}

.confidence-text {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}
</style>
