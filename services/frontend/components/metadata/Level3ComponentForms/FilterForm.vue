<!-- components/metadata/Level3ComponentForms/FilterForm.vue -->
<template>
  <div class="filter-form">
    <h3 class="text-lg font-semibold mb-4">Настройка фильтра: {{ componentId }}</h3>

    <div class="form-grid">
      <div class="form-group col-span-2">
        <label class="form-label required">Расположение</label>
        <select v-model="formData.location" class="form-select">
          <option value="suction">На входе (всасывающий)</option>
          <option value="pressure">На выходе насоса (напорный)</option>
          <option value="return">На возврате в резервуар (сливной)</option>
          <option value="pilot">На пилотной линии</option>
        </select>
      </div>

      <div class="form-group">
        <label class="form-label required">Тонкость фильтрации (микроны)</label>
        <select v-model.number="formData.filtration_rating" class="form-select">
          <option :value="10">10 мкм (очень тонкая)</option>
          <option :value="25">25 мкм (средняя)</option>
          <option :value="50">50 мкм (грубая)</option>
        </select>
        <p class="help-text">Для чувствительных компонентов: 10 мкм</p>
      </div>

      <div class="form-group">
        <label class="form-label required">Пропускная способность (л/мин)</label>
        <input v-model.number="formData.flow_capacity" type="number" class="form-input" placeholder="≥ Q_насоса" />
        <p class="help-text">Должна быть ≥ Q_насоса, иначе перепад велик</p>
      </div>

      <div class="form-group col-span-2 section-header">
        <h4 class="text-md font-semibold">История замены</h4>
      </div>

      <div class="form-group">
        <label class="form-label">Дата последней замены</label>
        <input v-model="formData.last_replacement" type="date" class="form-input" :max="today" />
        <p class="help-text text-amber-600">⚠ Грязный фильтр → кавитация</p>
      </div>

      <div class="form-group">
        <label class="form-label">Интервал замены (часы)</label>
        <input v-model.number="formData.replacement_interval_hours" type="number" class="form-input"
          placeholder="250-500 ч" />
        <p class="help-text">Обычно 250-500 ч в зависимости от условий</p>
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
import { useMetadataStore } from '~/stores/metadata';
import type { FilterSpecific } from '~/types/metadata';

const props = defineProps<{ componentId: string }>();
const store = useMetadataStore();

const component = computed(() => store.wizardState.system.components?.find(c => c.id === props.componentId));

const formData = reactive<FilterSpecific & any>({
  location: component.value?.filter_specific?.location || 'return',
  filtration_rating: component.value?.filter_specific?.filtration_rating || 25,
  flow_capacity: component.value?.filter_specific?.flow_capacity || undefined,
  last_replacement: component.value?.filter_specific?.last_replacement,
  replacement_interval_hours: component.value?.filter_specific?.replacement_interval_hours || 500,
});

const today = computed(() => new Date().toISOString().split('T')[0]);

const overallConfidence = computed(() => {
  let filled = 0;
  if (formData.location) filled++;
  if (formData.filtration_rating) filled++;
  if (formData.flow_capacity) filled++;
  return filled / 3;
});

const confidenceClass = computed(() => {
  if (overallConfidence.value < 0.5) return 'confidence-low';
  if (overallConfidence.value < 0.7) return 'confidence-medium';
  return 'confidence-high';
});

function updateStore() {
  store.updateComponent(props.componentId, {
    filter_specific: {
      location: formData.location,
      filtration_rating: formData.filtration_rating,
      flow_capacity: formData.flow_capacity!,
      last_replacement: formData.last_replacement,
      replacement_interval_hours: formData.replacement_interval_hours,
    },
    confidence_scores: { overall: overallConfidence.value }
  });
}

watch(formData, updateStore, { deep: true });
</script>

<style scoped>
.filter-form {
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
