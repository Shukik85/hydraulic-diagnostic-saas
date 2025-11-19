<!-- components/metadata/Level3ComponentForms/AccumulatorForm.vue -->
<template>
  <div class="accumulator-form">
    <h3 class="text-lg font-semibold mb-4">Настройка аккумулятора: {{ componentId }}</h3>

    <div class="form-grid">
      <div class="form-group col-span-2">
        <label class="form-label required">Тип аккумулятора</label>
        <select v-model="formData.accumulator_type" class="form-select">
          <option value="hydropneumatic">Гидропневматический (воздух + жидкость)</option>
          <option value="piston">Поршневой</option>
          <option value="diaphragm">Мембранный</option>
          <option value="spring">Пружинный</option>
        </select>
      </div>

      <div class="form-group">
        <label class="form-label required">Рабочий объём (литры)</label>
        <input v-model.number="formData.volume" type="number" class="form-input" placeholder="1-50 л" />
        <p class="help-text">Хранит энергию под давлением</p>
      </div>

      <div class="form-group">
        <label class="form-label required">Максимальное давление (бар)</label>
        <input v-model.number="formData.max_pressure" type="number" class="form-input"
          placeholder="Обычно = P_max системы" />
      </div>

      <div class="form-group">
        <label class="form-label required">Давление предзарядки (бар)</label>
        <input v-model.number="formData.precharge_pressure" type="number" class="form-input"
          placeholder="Обычно 0.6 × P_max" />
        <p class="help-text">Критично для компенсации колебаний</p>
      </div>

      <div class="form-group col-span-2">
        <label class="form-label">Функция в системе</label>
        <select v-model="formData.function" class="form-select">
          <option value="pulsation_dampening">Компенсация колебаний (сглаживание пиков)</option>
          <option value="emergency_power">Аварийное питание (отключение при потере)</option>
          <option value="energy_recovery">Восстановление энергии (рекуперация)</option>
          <option value="shock_absorption">Гашение гидроударов</option>
        </select>
      </div>

      <div class="form-group col-span-2 section-header">
        <h4 class="text-md font-semibold">Состояние азотной подвески</h4>
      </div>

      <div class="form-group">
        <label class="form-label">Дата последней проверки</label>
        <input v-model="formData.nitrogen_check_date" type="date" class="form-input" :max="today" />
        <p class="help-text text-red-600" v-if="!formData.nitrogen_check_date">
          ⚠ Никогда не проверялось → срочно проверить!
        </p>
      </div>

      <div class="form-group">
        <label class="form-label">Периодичность проверки (месяцы)</label>
        <input v-model.number="formData.check_interval_months" type="number" class="form-input"
          placeholder="6-12 месяцев" />
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
import type { AccumulatorSpecific } from '~/types/metadata';

const props = defineProps<{ componentId: string }>();
const store = useMetadataStore();

const component = computed(() => store.wizardState.system.components?.find(c => c.id === props.componentId));

const formData = reactive<AccumulatorSpecific & any>({
  accumulator_type: component.value?.accumulator_specific?.accumulator_type || 'hydropneumatic',
  volume: component.value?.accumulator_specific?.volume || undefined,
  max_pressure: component.value?.max_pressure,
  precharge_pressure: component.value?.accumulator_specific?.precharge_pressure || undefined,
  function: component.value?.accumulator_specific?.function || 'pulsation_dampening',
  nitrogen_check_date: component.value?.accumulator_specific?.nitrogen_check_date,
  check_interval_months: 12,
});

const today = computed(() => new Date().toISOString().split('T')[0]);

const overallConfidence = computed(() => {
  let filled = 0;
  if (formData.accumulator_type) filled++;
  if (formData.volume) filled++;
  if (formData.max_pressure) filled++;
  if (formData.precharge_pressure) filled++;
  return filled / 4;
});

const confidenceClass = computed(() => {
  if (overallConfidence.value < 0.5) return 'confidence-low';
  if (overallConfidence.value < 0.7) return 'confidence-medium';
  return 'confidence-high';
});

function updateStore() {
  store.updateComponent(props.componentId, {
    accumulator_specific: {
      accumulator_type: formData.accumulator_type,
      volume: formData.volume!,
      precharge_pressure: formData.precharge_pressure!,
      function: formData.function,
      nitrogen_check_date: formData.nitrogen_check_date,
    },
    max_pressure: formData.max_pressure,
    confidence_scores: { overall: overallConfidence.value }
  });
}

watch(formData, updateStore, { deep: true });
</script>

<style scoped>
.accumulator-form {
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
