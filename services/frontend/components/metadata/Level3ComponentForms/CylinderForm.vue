<!-- components/metadata/Level3ComponentForms/CylinderForm.vue -->
<template>
  <div class="cylinder-form">
    <h3 class="text-lg font-semibold mb-4">Настройка гидроцилиндра: {{ componentId }}</h3>

    <div class="form-grid">
      <!-- Функция цилиндра -->
      <div class="form-group col-span-2">
        <label class="form-label required">Функция цилиндра</label>
        <select v-model="formData.function" class="form-select">
          <option value="primary">Основной исполнительный (boom, stick)</option>
          <option value="auxiliary">Вспомогательный (bucket tilt, rotation)</option>
          <option value="balancing">Балансировочный (удержание без дрейфа)</option>
          <option value="cable">Кабельный (натяжение)</option>
        </select>
      </div>

      <!-- Диаметр поршня -->
      <div class="form-group">
        <label class="form-label required">Диаметр поршня (мм)</label>
        <input v-model.number="formData.piston_diameter" type="number" class="form-input" placeholder="30-300 мм"
          min="30" max="300" />
        <p class="help-text">Влияет на развиваемое усилие: F = P × π × D² / 4</p>
      </div>

      <!-- Ход поршня -->
      <div class="form-group">
        <label class="form-label required">Ход поршня (мм)</label>
        <input v-model.number="formData.stroke_length" type="number" class="form-input" placeholder="100-2000 мм"
          min="100" max="2000" />
        <p class="help-text">Максимальное линейное перемещение</p>
      </div>

      <!-- Площадь штока (опционально) -->
      <div class="form-group">
        <label class="form-label">Площадь штока (мм²)</label>
        <input v-model.number="formData.rod_area" type="number" class="form-input"
          placeholder="Автоматически если известен диаметр" />
      </div>

      <!-- Максимальное давление -->
      <div class="form-group">
        <label class="form-label required">Максимальное давление (бар)</label>
        <input v-model.number="formData.max_pressure" type="number" class="form-input" placeholder="80-350 бар" min="80"
          max="350" />
      </div>

      <!-- Критический тип отказа -->
      <div class="form-group col-span-2">
        <label class="form-label">Критический тип отказа</label>
        <select v-model="formData.failure_mode" class="form-select">
          <option value="internal_leak">Внутренние утечки (износ поршня)</option>
          <option value="external_leak">Внешние утечки (разрушение уплотнений)</option>
          <option value="seizure">Заклинивание (грязь, коррозия)</option>
          <option value="rupture">Потеря герметичности (разрыв цилиндра)</option>
        </select>
      </div>

      <!-- Характер движения -->
      <div class="form-group col-span-2">
        <label class="form-label">Характер движения</label>
        <select v-model="formData.movement_character" class="form-select">
          <option value="smooth">Плавное управляемое движение</option>
          <option value="fast_switching">Быстрое переключение (вкл/выкл)</option>
          <option value="pulsed">Импульсное (часто вкл/выкл)</option>
        </select>
      </div>

      <!-- Нормальная температура -->
      <div class="form-group col-span-2">
        <label class="form-label">Нормальная температура работы (°C)</label>
        <div class="range-input">
          <input v-model.number="formData.normal_temp_min" type="number" class="form-input" placeholder="Мин" />
          <span class="range-separator">—</span>
          <input v-model.number="formData.normal_temp_max" type="number" class="form-input" placeholder="Макс" />
        </div>
        <p class="help-text text-amber-600">⚠ Критично: > 80°C — риск деградации уплотнений</p>
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
        <input v-model.number="formData.operating_hours" type="number" class="form-input" />
      </div>
    </div>

    <!-- Confidence indicator -->
    <div class="confidence-indicator" :class="confidenceClass">
      <div class="confidence-bar">
        <div class="confidence-fill" :style="{ width: `${overallConfidence * 100}%` }"></div>
      </div>
      <p class="confidence-text">Полнота данных: {{ Math.round(overallConfidence * 100) }}%</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata';
import type { CylinderSpecific } from '~/types/metadata';

const props = defineProps<{ componentId: string }>();
const store = useMetadataStore();

const component = computed(() =>
  store.wizardState.system.components?.find(c => c.id === props.componentId)
);

const formData = reactive<CylinderSpecific & any>({
  function: component.value?.cylinder_specific?.function || 'primary',
  piston_diameter: component.value?.cylinder_specific?.piston_diameter || undefined,
  stroke_length: component.value?.cylinder_specific?.stroke_length || undefined,
  rod_area: component.value?.cylinder_specific?.rod_area,
  failure_mode: component.value?.cylinder_specific?.failure_mode || 'internal_leak',
  movement_character: component.value?.cylinder_specific?.movement_character || 'smooth',
  max_pressure: component.value?.max_pressure,
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

  if (formData.function) filled++;
  if (formData.piston_diameter) filled++;
  if (formData.stroke_length) filled++;
  if (formData.max_pressure) filled++;

  return filled / total;
});

const confidenceClass = computed(() => {
  if (overallConfidence.value < 0.5) return 'confidence-low';
  if (overallConfidence.value < 0.7) return 'confidence-medium';
  return 'confidence-high';
});

function updateStore() {
  store.updateComponent(props.componentId, {
    cylinder_specific: {
      function: formData.function,
      piston_diameter: formData.piston_diameter!,
      stroke_length: formData.stroke_length!,
      rod_area: formData.rod_area,
      failure_mode: formData.failure_mode,
      movement_character: formData.movement_character,
    },
    max_pressure: formData.max_pressure,
    normal_ranges: {
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
.cylinder-form {
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
