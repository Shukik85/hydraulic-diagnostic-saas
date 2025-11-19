<!-- components/metadata/Level3ComponentForms/ValveForm.vue -->
<template>
  <div class="valve-form">
    <h3 class="text-lg font-semibold mb-4">Настройка клапана: {{ componentId }}</h3>

    <div class="form-grid">
      <div class="form-group col-span-2">
        <label class="form-label required">Тип клапана</label>
        <select v-model="formData.valve_type" class="form-select">
          <option value="relief">Предохранительный (обеспечивает P_max)</option>
          <option value="directional">Направляющий (распределитель)</option>
          <option value="check">Обратный (пропускает в одном направлении)</option>
          <option value="throttle">Дроссельный (регулирует расход)</option>
          <option value="reducing">Редукционный (поддерживает давление)</option>
        </select>
      </div>

      <div class="form-group">
        <label class="form-label required">Номинальный расход (л/мин)</label>
        <input v-model.number="formData.nominal_flow_rate" type="number" class="form-input"
          placeholder="10-500 л/мин" />
        <p class="help-text">Должен быть ≥ расхода через клапан</p>
      </div>

      <div class="form-group" v-if="formData.valve_type === 'relief' || formData.valve_type === 'reducing'">
        <label class="form-label">Уставка давления (бар)</label>
        <input v-model.number="formData.pressure_setpoint" type="number" class="form-input"
          placeholder="Для предохр. = P_max" />
      </div>

      <div class="form-group col-span-2">
        <label class="form-label">Состояние</label>
        <select v-model="formData.state" class="form-select">
          <option value="operational">Исправен</option>
          <option value="needs_adjustment">Требует регулировки</option>
          <option value="under_replacement">В процессе замены</option>
          <option value="unknown">Неизвестно</option>
        </select>
      </div>

      <div class="form-group col-span-2 section-header">
        <h4 class="text-md font-semibold">История обслуживания</h4>
      </div>

      <div class="form-group">
        <label class="form-label">Дата последнего ТО</label>
        <input v-model="formData.last_maintenance" type="date" class="form-input" :max="today" />
      </div>

      <div class="form-group">
        <label class="form-label">Периодичность ТО (часы)</label>
        <input v-model.number="formData.maintenance_interval_hours" type="number" class="form-input" />
      </div>
    </div>

    <div class="confidence-indicator" :class="confidenceClass">
      <div class="confidence-bar">
        <div class="confidence-fill" :style="{ width: `${overallConfidence * 100}%` }"></div>
      </div>
      <p class="confidence-text">Полнота данных: {{ Math.round(overallConfidence * 100) }}%</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'

import { useMetadataStore } from '~/stores/metadata';
import type { ValveSpecific } from '~/types/metadata';

const props = defineProps<{ componentId: string }>();
const store = useMetadataStore();

const component = computed(() => store.wizardState.system.components?.find(c => c.id === props.componentId));

const formData = reactive<ValveSpecific & any>({
  valve_type: component.value?.valve_specific?.valve_type || 'directional',
  nominal_flow_rate: component.value?.valve_specific?.nominal_flow_rate || undefined,
  pressure_setpoint: component.value?.valve_specific?.pressure_setpoint,
  state: component.value?.valve_specific?.state || 'operational',
  last_maintenance: component.value?.last_maintenance,
  maintenance_interval_hours: component.value?.maintenance_interval_hours,
});

const today = computed(() => new Date().toISOString().split('T')[0]);

const overallConfidence = computed(() => {
  let filled = 0;
  if (formData.valve_type) filled++;
  if (formData.nominal_flow_rate) filled++;
  return filled / 2;
});

const confidenceClass = computed(() => {
  if (overallConfidence.value < 0.5) return 'confidence-low';
  if (overallConfidence.value < 0.7) return 'confidence-medium';
  return 'confidence-high';
});

function updateStore() {
  store.updateComponent(props.componentId, {
    valve_specific: {
      valve_type: formData.valve_type,
      nominal_flow_rate: formData.nominal_flow_rate!,
      pressure_setpoint: formData.pressure_setpoint,
      state: formData.state,
    },
    last_maintenance: formData.last_maintenance,
    maintenance_interval_hours: formData.maintenance_interval_hours,
    confidence_scores: { overall: overallConfidence.value }
  });
}

watch(formData, updateStore, { deep: true });
</script>

<style scoped>
.valve-form {
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

.help-text {
  font-size: 0.75rem;
  color: #6b7280;
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
