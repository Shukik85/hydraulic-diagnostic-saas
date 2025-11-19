<!-- components/metadata/Level3ComponentForms.vue -->
<template>
  <div class="level-3">
    <h2 class="text-xl font-semibold mb-4">3. Характеристики компонентов</h2>

    <p v-if="store.componentsCount === 0" class="text-gray-600 mb-4">
      Вернитесь на Уровень 2 и добавьте компоненты на схему.
    </p>

    <div v-else class="components-forms">
      <!-- Выбор компонента -->
      <div class="component-selector mb-6">
        <label class="form-label">Выберите компонент для настройки:</label>
        <select v-model="selectedComponentId" class="form-select">
          <option value="">-- Выберите компонент --</option>
          <option v-for="comp in store.wizardState.system.components" :key="comp.id" :value="comp.id">
            {{ getComponentLabel(comp) }}
          </option>
        </select>
      </div>

      <!-- Динамическая форма -->
      <transition name="fade" mode="out-in">
        <component v-if="selectedComponentId && currentFormComponent" :is="currentFormComponent"
          :component-id="selectedComponentId" :key="selectedComponentId" />
        <div v-else class="empty-state">
          <p class="text-gray-500">Выберите компонент для настройки его характеристик</p>
        </div>
      </transition>
    </div>

    <!-- Progress indicator -->
    <div class="progress-summary mt-6">
      <h3 class="text-sm font-semibold mb-2">Заполненность компонентов:</h3>
      <div class="components-progress">
        <div v-for="comp in store.wizardState.system.components" :key="comp.id" class="component-progress-item">
          <span class="component-name">{{ comp.id }}</span>
          <div class="progress-bar-sm">
            <div class="progress-fill-sm" :style="{ width: `${getComponentCompleteness(comp) * 100}%` }"></div>
          </div>
          <span class="progress-pct">{{ Math.round(getComponentCompleteness(comp) * 100) }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed } from 'vue'

import { useMetadataStore } from '~/stores/metadata';
import type { ComponentMetadata, ComponentType } from '~/types/metadata';
import PumpForm from '~/components/metadata/Level3ComponentForms/PumpForm.vue';
import MotorForm from '~/components/metadata/Level3ComponentForms/MotorForm.vue';
import CylinderForm from '~/components/metadata/Level3ComponentForms/CylinderForm.vue';
import ValveForm from '~/components/metadata/Level3ComponentForms/ValveForm.vue';
import FilterForm from '~/components/metadata/Level3ComponentForms/FilterForm.vue';
import AccumulatorForm from '~/components/metadata/Level3ComponentForms/AccumulatorForm.vue';


const store = useMetadataStore();

const selectedComponentId = ref<string>('');

// Auto-select first component
onMounted(() => {
  const firstComponent = store.wizardState.system.components?.[0];
  if (firstComponent) {
    selectedComponentId.value = firstComponent.id;
  }
});


const selectedComponent = computed(() =>
  store.wizardState.system.components?.find(c => c.id === selectedComponentId.value)
);

const currentFormComponent = computed(() => {
  if (!selectedComponent.value) return null;

  const formComponents: Record<ComponentType, any> = {
    pump: PumpForm,
    motor: MotorForm,
    cylinder: CylinderForm,
    valve: ValveForm,
    filter: FilterForm,
    accumulator: AccumulatorForm,
  };

  return formComponents[selectedComponent.value.component_type];
});

function getComponentLabel(comp: ComponentMetadata): string {
  const typeLabels: Record<ComponentType, string> = {
    pump: '⚙️ Насос',
    motor: '🔄 Мотор',
    cylinder: '⬌ Цилиндр',
    valve: '⬥ Клапан',
    filter: '◈ Фильтр',
    accumulator: '⬢ Аккумулятор'
  };
  return `${typeLabels[comp.component_type]} — ${comp.id}`;
}

function getComponentCompleteness(comp: ComponentMetadata): number {
  let filled = 0;
  let total = 5;

  if (comp.max_pressure) filled++;
  if (comp.normal_ranges.pressure) filled++;
  if (comp.normal_ranges.temperature) filled++;

  // Специфичные поля
  if (comp.component_type === 'pump' && comp.pump_specific?.nominal_flow_rate) filled++;
  if (comp.component_type === 'motor' && comp.motor_specific?.displacement) filled++;
  if (comp.component_type === 'cylinder' && comp.cylinder_specific?.piston_diameter) filled++;

  // История
  if (comp.last_maintenance) filled++;

  return filled / total;
}
</script>

<style scoped>
.level-3 {
  padding: 1rem;
}

.component-selector {
  max-width: 500px;
}

.form-label {
  font-weight: 500;
  font-size: 0.875rem;
  color: #374151;
  margin-bottom: 0.5rem;
  display: block;
}

.form-select {
  width: 100%;
  padding: 0.625rem 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
}

.form-select:focus {
  outline: none;
  border-color: #3b82f6;
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

.empty-state {
  padding: 3rem;
  text-align: center;
  background: #f9fafb;
  border: 2px dashed #d1d5db;
  border-radius: 0.75rem;
}

.progress-summary {
  padding: 1rem;
  background: #f9fafb;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
}

.components-progress {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.component-progress-item {
  display: grid;
  grid-template-columns: 150px 1fr 60px;
  align-items: center;
  gap: 1rem;
}

.component-name {
  font-size: 0.875rem;
  color: #374151;
  font-weight: 500;
}

.progress-bar-sm {
  height: 6px;
  background: #e5e7eb;
  border-radius: 3px;
  overflow: hidden;
}

.progress-fill-sm {
  height: 100%;
  background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
  transition: width 0.3s;
}

.progress-pct {
  font-size: 0.75rem;
  color: #6b7280;
  text-align: right;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
