<!-- components/metadata/Level3ComponentForms/PumpForm.vue -->
<template>
  <div class="pump-form">
    <h3 class="text-lg font-semibold mb-4">Настройка насоса: {{ componentId }}</h3>

    <div class="form-grid">
      <!-- Тип насоса -->
      <div class="form-group col-span-2">
        <label class="form-label required">Тип насоса</label>
        <select v-model="formData.pump_type" class="form-select" @change="onPumpTypeChange">
          <option value="">Выберите тип</option>
          <option value="axial_piston">Аксиально-поршневой (переменная производительность)</option>
          <option value="gear">Шестеренчатый (постоянная производительность)</option>
          <option value="vane">Пластинчатый</option>
          <option value="radial_piston">Радиально-поршневой</option>
        </select>
        <p v-if="formData.pump_type" class="help-text">
          {{ getPumpTypeDescription(formData.pump_type) }}
        </p>
      </div>

      <!-- Номинальная производительность -->
      <div class="form-group">
        <label class="form-label required">Номинальная производительность (л/мин)</label>
        <input v-model.number="formData.nominal_flow_rate" type="number" class="form-input"
          :placeholder="getFlowRatePlaceholder()" min="10" max="500" />
        <div class="smart-default" v-if="smartDefaults.nominal_flow_rate">
          <button @click="applyDefault('nominal_flow_rate')" class="default-btn">
            Рекомендуемое: {{ smartDefaults.nominal_flow_rate }} л/мин
          </button>
          <span class="confidence">Confidence: {{ (smartDefaults.confidence * 100).toFixed(0) }}%</span>
        </div>
      </div>

      <!-- Максимальное давление -->
      <div class="form-group">
        <label class="form-label required">Максимальное давление (бар)</label>
        <input v-model.number="formData.max_pressure" type="number" class="form-input" placeholder="80-350 бар" min="80"
          max="350" />
        <div class="smart-default" v-if="smartDefaults.max_pressure">
          <button @click="applyDefault('max_pressure')" class="default-btn">
            Типично для {{ equipmentType }}: {{ smartDefaults.max_pressure }} бар
          </button>
          <span class="confidence">Confidence: {{ (smartDefaults.confidence * 100).toFixed(0) }}%</span>
        </div>
      </div>

      <!-- Объемный КПД -->
      <div class="form-group">
        <label class="form-label">Объемный КПД (η_об)</label>
        <input v-model.number="formData.volumetric_efficiency" type="number" class="form-input" placeholder="0.70-0.99"
          min="0.7" max="0.99" step="0.01" />
        <div class="smart-default" v-if="smartDefaults.volumetric_efficiency">
          <button @click="applyDefault('volumetric_efficiency')" class="default-btn">
            Для нового {{ formData.pump_type }}: {{ smartDefaults.volumetric_efficiency }}
          </button>
        </div>
        <p class="help-text">Если не указан, будет установлено 0.96 по умолчанию</p>
      </div>

      <!-- Механический КПД -->
      <div class="form-group">
        <label class="form-label">Механический КПД (η_мех)</label>
        <input v-model.number="formData.mechanical_efficiency" type="number" class="form-input" placeholder="0.80-0.99"
          min="0.8" max="0.99" step="0.01" />
        <div class="smart-default" v-if="smartDefaults.mechanical_efficiency">
          <button @click="applyDefault('mechanical_efficiency')" class="default-btn">
            Типично: {{ smartDefaults.mechanical_efficiency }}
          </button>
        </div>
      </div>

      <!-- Условные поля для аксиально-поршневого -->
      <template v-if="formData.pump_type === 'axial_piston'">
        <div class="form-group col-span-2">
          <label class="form-label">Тип регулирования</label>
          <select v-model="formData.regulation_type" class="form-select">
            <option value="">Без регулирования (fixed displacement)</option>
            <option value="pressure_compensator">По давлению (compensator)</option>
            <option value="load_sensing">По нагрузке (load-sensing)</option>
          </select>
          <p class="help-text">
            Load-sensing обеспечивает лучшую энергоэффективность
          </p>
        </div>

        <div class="form-group">
          <label class="form-label">Максимальный наклон (°)</label>
          <input v-model.number="formData.max_swash_angle" type="number" class="form-input" placeholder="0-30°" min="0"
            max="30" />
          <p class="help-text">Определяет точность управления производительностью</p>
        </div>
      </template>

      <!-- Нормальный диапазон давления -->
      <div class="form-group col-span-2">
        <label class="form-label">Нормальный диапазон давления (бар)</label>
        <div class="range-input">
          <input v-model.number="formData.normal_pressure_min" type="number" class="form-input" placeholder="Мин"
            :max="formData.normal_pressure_max" />
          <span class="range-separator">—</span>
          <input v-model.number="formData.normal_pressure_max" type="number" class="form-input" placeholder="Макс"
            :min="formData.normal_pressure_min" />
        </div>
        <div class="smart-default" v-if="formData.max_pressure && !formData.normal_pressure_min">
          <button @click="applyPressureRange" class="default-btn">
            Рекомендуется: [{{ Math.round(formData.max_pressure * 0.5) }}, {{ Math.round(formData.max_pressure * 0.85)
            }}] бар
          </button>
          <span class="confidence">Confidence: 70%</span>
        </div>
      </div>

      <!-- Нормальный диапазон температуры -->
      <div class="form-group col-span-2">
        <label class="form-label">Нормальная температура работы (°C)</label>
        <div class="range-input">
          <input v-model.number="formData.normal_temp_min" type="number" class="form-input" placeholder="Мин"
            :max="formData.normal_temp_max" />
          <span class="range-separator">—</span>
          <input v-model.number="formData.normal_temp_max" type="number" class="form-input" placeholder="Макс"
            :min="formData.normal_temp_min" />
        </div>
        <div class="smart-default" v-if="!formData.normal_temp_min">
          <button @click="applyTempRange" class="default-btn">
            Типично для насосов: [40, 70] °C
          </button>
          <span class="confidence">Confidence: 75%</span>
        </div>
        <p class="help-text text-amber-600">⚠ Критично: > 80°C ведёт к утечкам и деградации масла</p>
      </div>

      <!-- Максимально допустимая вибрация -->
      <div class="form-group">
        <label class="form-label">Макс. допустимая вибрация (м/с²)</label>
        <input v-model.number="formData.max_vibration" type="number" class="form-input" placeholder="0.5-3.0 м/с²"
          min="0.5" max="10" step="0.1" />
        <p class="help-text">Норма: 0.5-3.0 м/с² (ISO 13849)</p>
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
          placeholder="500-1000 ч" min="100" max="5000" />
        <p class="help-text">Обычно 500-1000 часов для строительной техники</p>
      </div>

      <div class="form-group">
        <label class="form-label">Текущая наработка (часы)</label>
        <input v-model.number="formData.operating_hours" type="number" class="form-input"
          placeholder="Примерная наработка" min="0" />
      </div>
    </div>

    <!-- Confidence indicator -->
    <div class="confidence-indicator" :class="confidenceClass">
      <div class="confidence-bar">
        <div class="confidence-fill" :style="{ width: `${overallConfidence * 100}%` }"></div>
      </div>
      <p class="confidence-text">
        Полнота данных: {{ Math.round(overallConfidence * 100) }}%
        <span v-if="overallConfidence < 0.5" class="text-red-600">— требуется больше данных</span>
        <span v-else-if="overallConfidence < 0.7" class="text-amber-600">— хорошо для типичной системы</span>
        <span v-else class="text-green-600">— отлично, высокая надёжность</span>
      </p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'

import { useMetadataStore } from '~/stores/metadata';
import type { PumpSpecific } from '~/types/metadata';

const props = defineProps<{
  componentId: string;
}>();

const store = useMetadataStore();

const component = computed(() =>
  store.wizardState.system.components?.find(c => c.id === props.componentId)
);

const equipmentType = computed(() => store.wizardState.system.equipment_type || 'excavator_tracked');

const formData = reactive<PumpSpecific & {
  max_pressure?: number;
  normal_pressure_min?: number;
  normal_pressure_max?: number;
  normal_temp_min?: number;
  normal_temp_max?: number;
  max_vibration?: number;
  last_maintenance?: string;
  maintenance_interval_hours?: number;
  operating_hours?: number;
}>({
  pump_type: component.value?.pump_specific?.pump_type || '' as any,
  nominal_flow_rate: component.value?.pump_specific?.nominal_flow_rate || undefined,
  max_pressure: component.value?.max_pressure || undefined,
  volumetric_efficiency: component.value?.pump_specific?.volumetric_efficiency || undefined,
  mechanical_efficiency: component.value?.pump_specific?.mechanical_efficiency || undefined,
  regulation_type: component.value?.pump_specific?.regulation_type || undefined,
  max_swash_angle: component.value?.pump_specific?.max_swash_angle || undefined,
  normal_pressure_min: component.value?.normal_ranges?.pressure?.min,
  normal_pressure_max: component.value?.normal_ranges?.pressure?.max,
  normal_temp_min: component.value?.normal_ranges?.temperature?.min,
  normal_temp_max: component.value?.normal_ranges?.temperature?.max,
  max_vibration: component.value?.normal_ranges?.vibration?.max,
  last_maintenance: component.value?.last_maintenance,
  maintenance_interval_hours: component.value?.maintenance_interval_hours,
  operating_hours: component.value?.operating_hours,
});

const today = computed(() => new Date().toISOString().split('T')[0]);

// Smart Defaults
const smartDefaults = computed(() => {
  const defaults: any = { confidence: 0.7 };

  // Расход зависит от типа оборудования
  const flowRates: Record<string, number> = {
    excavator_tracked: 150,
    excavator_wheeled: 120,
    loader_wheeled: 100,
    crane_mobile: 80,
  };
  defaults.nominal_flow_rate = flowRates[equipmentType.value] || 120;

  // Давление типично 210-250 бар для строительной техники
  defaults.max_pressure = 210;

  // КПД зависит от типа насоса
  if (formData.pump_type === 'axial_piston') {
    defaults.volumetric_efficiency = 0.96;
    defaults.mechanical_efficiency = 0.94;
    defaults.confidence = 0.75;
  } else if (formData.pump_type === 'gear') {
    defaults.volumetric_efficiency = 0.92;
    defaults.mechanical_efficiency = 0.90;
    defaults.confidence = 0.7;
  }

  return defaults;
});

// Overall confidence (заполнено ли достаточно полей)
const overallConfidence = computed(() => {
  let filled = 0;
  let total = 5; // критичные поля: type, flow_rate, max_pressure, normal_pressure, normal_temp

  if (formData.pump_type) filled++;
  if (formData.nominal_flow_rate) filled++;
  if (formData.max_pressure) filled++;
  if (formData.normal_pressure_min && formData.normal_pressure_max) filled++;
  if (formData.normal_temp_min && formData.normal_temp_max) filled++;

  return filled / total;
});

const confidenceClass = computed(() => {
  if (overallConfidence.value < 0.5) return 'confidence-low';
  if (overallConfidence.value < 0.7) return 'confidence-medium';
  return 'confidence-high';
});

function getPumpTypeDescription(type: string): string {
  const descriptions: Record<string, string> = {
    axial_piston: 'Высокая эффективность, переменная производительность, регулируемое давление',
    gear: 'Простая конструкция, постоянная производительность, надёжность',
    vane: 'Низкий уровень шума, плавная работа, средняя производительность',
    radial_piston: 'Высокое давление, низкая скорость, высокая точность'
  };
  return descriptions[type] || '';
}

function getFlowRatePlaceholder(): string {
  const ranges: Record<string, string> = {
    excavator_tracked: '80-200 л/мин (20-30т)',
    excavator_wheeled: '60-150 л/мин',
    loader_wheeled: '50-120 л/мин',
    crane_mobile: '40-100 л/мин',
  };
  return ranges[equipmentType.value] || '10-500 л/мин';
}

function applyDefault(field: keyof typeof formData) {
  (formData as any)[field] = smartDefaults.value[field];
}

function applyPressureRange() {
  if (formData.max_pressure) {
    formData.normal_pressure_min = Math.round(formData.max_pressure * 0.5);
    formData.normal_pressure_max = Math.round(formData.max_pressure * 0.85);
  }
}

function applyTempRange() {
  formData.normal_temp_min = 40;
  formData.normal_temp_max = 70;
}

function onPumpTypeChange() {
  // Сбросить условные поля при смене типа
  if (formData.pump_type !== 'axial_piston') {
    formData.regulation_type = undefined;
    formData.max_swash_angle = undefined;
  }
  updateStore();
}

function updateStore() {
  store.updateComponent(props.componentId, {
    pump_specific: {
      pump_type: formData.pump_type,
      nominal_flow_rate: formData.nominal_flow_rate!,
      volumetric_efficiency: formData.volumetric_efficiency,
      mechanical_efficiency: formData.mechanical_efficiency,
      regulation_type: formData.regulation_type,
      max_swash_angle: formData.max_swash_angle,
    },
    max_pressure: formData.max_pressure,
    normal_ranges: {
      pressure: formData.normal_pressure_min && formData.normal_pressure_max
        ? { min: formData.normal_pressure_min, max: formData.normal_pressure_max, unit: 'bar' }
        : undefined,
      temperature: formData.normal_temp_min && formData.normal_temp_max
        ? { min: formData.normal_temp_min, max: formData.normal_temp_max, unit: '°C' }
        : undefined,
      vibration: formData.max_vibration
        ? { min: 0, max: formData.max_vibration, unit: 'м/с²' }
        : undefined,
    },
    last_maintenance: formData.last_maintenance,
    maintenance_interval_hours: formData.maintenance_interval_hours,
    operating_hours: formData.operating_hours,
    confidence_scores: {
      overall: overallConfidence.value,
      smart_defaults: smartDefaults.value.confidence
    }
  });
}

watch(formData, () => {
  updateStore();
}, { deep: true });
</script>

<style scoped>
.pump-form {
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
  transition: border-color 0.2s;
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
  align-items: center;
  gap: 0.5rem;
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
  transition: background 0.2s;
  text-align: left;
}

.default-btn:hover {
  background: #2563eb;
}

.confidence {
  font-size: 0.65rem;
  color: #6b7280;
  white-space: nowrap;
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
