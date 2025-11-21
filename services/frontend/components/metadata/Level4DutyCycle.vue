<!-- components/metadata/Level4DutyCycle.vue -->
<template>
  <div class="level-4">
    <h2 class="text-xl font-semibold mb-4">4. Профиль нагрузки и условия эксплуатации</h2>

    <p class="text-gray-600 mb-6">
      Опишите типичные режимы работы оборудования. Это поможет модели учесть специфику нагрузок.
    </p>

    <div class="form-sections">
      <!-- Тип профиля -->
      <div class="form-section">
        <h3 class="section-title">Типичный профиль работы</h3>

        <div class="profile-selector">
          <button v-for="profile in profiles" :key="profile.type" @click="selectProfile(profile.type)"
            :class="['profile-card', { selected: formData.profile_type === profile.type }]">
            <span class="profile-icon">{{ profile.icon }}</span>
            <div class="profile-info">
              <div class="profile-name">{{ profile.name }}</div>
              <div class="profile-description">{{ profile.description }}</div>
            </div>
          </button>
        </div>
      </div>

      <!-- Распределение нагрузки (если выбран профиль) -->
      <div v-if="formData.profile_type" class="form-section">
        <h3 class="section-title">Распределение нагрузки</h3>
        <p class="text-sm text-gray-600 mb-4">
          Укажите примерное процентное распределение времени работы по типам операций
        </p>

        <div class="load-distribution">
          <div v-for="(value, key) in formData.load_distribution" :key="key" class="distribution-item">
            <label class="distribution-label">{{ getLoadLabel(key) }}</label>
            <div class="distribution-input">
              <input v-model.number="formData.load_distribution[key]" type="number" class="form-input" min="0"
                max="100" />
              <span class="distribution-unit">%</span>
            </div>
          </div>
        </div>

        <div class="total-indicator" :class="{ error: loadTotal !== 100 }">
          Итого: {{ loadTotal }}%
          <span v-if="loadTotal !== 100" class="error-text">(должно быть 100%)</span>
        </div>
      </div>

      <!-- Частота пиковых нагрузок -->
      <div class="form-section">
        <h3 class="section-title">Частота пиковых нагрузок</h3>
        <select v-model="formData.peak_load_frequency" class="form-select">
          <option value="rare">Редко (< 10% времени)</option>
          <option value="regular">Регулярно (10-30% времени)</option>
          <option value="frequent">Часто (30-50% времени)</option>
          <option value="constant">Постоянно (> 50% времени)</option>
        </select>
        <p class="help-text">
          {{ getPeakLoadDescription(formData.peak_load_frequency) }}
        </p>
      </div>

      <!-- Интервалы перерывов -->
      <div class="form-section">
        <h3 class="section-title">Среднее время между перерывами</h3>
        <div class="slider-container">
          <input v-model.number="formData.break_interval_minutes" type="range" min="5" max="60" step="5"
            class="slider" />
          <span class="slider-value">{{ formData.break_interval_minutes }} минут</span>
        </div>
        <p class="help-text">
          Влияет на теплоотвод: частые перерывы = лучше охлаждается
        </p>
      </div>

      <!-- Условия эксплуатации -->
      <div class="form-section">
        <h3 class="section-title">Условия окружающей среды</h3>

        <div class="conditions-grid">
          <div class="condition-item">
            <label class="condition-label">Температура (мин, °C)</label>
            <input v-model.number="formData.ambient_conditions.temp_min" type="number" class="form-input"
              placeholder="-30" />
          </div>

          <div class="condition-item">
            <label class="condition-label">Температура (макс, °C)</label>
            <input v-model.number="formData.ambient_conditions.temp_max" type="number" class="form-input"
              placeholder="+50" />
          </div>
        </div>

        <div class="checkboxes">
          <label class="checkbox-item">
            <input v-model="formData.ambient_conditions.dusty" type="checkbox" />
            <span>Пыльная/грязная среда (повышенный износ)</span>
          </label>

          <label class="checkbox-item">
            <input v-model="formData.ambient_conditions.humid" type="checkbox" />
            <span>Влажная среда (коррозия)</span>
          </label>

          <label class="checkbox-item">
            <input v-model="formData.ambient_conditions.high_vibration" type="checkbox" />
            <span>Высокие вибрации (близость к дороге/механизму)</span>
          </label>

          <label class="checkbox-item">
            <input v-model="formData.ambient_conditions.hot_environment" type="checkbox" />
            <span>Горячая среда (солнце, близость к источникам тепла)</span>
          </label>

          <label class="checkbox-item">
            <input v-model="formData.ambient_conditions.high_altitude" type="checkbox" />
            <span>Высокогорье (проблемы с охлаждением)</span>
          </label>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'

import { useMetadataStore } from '~/stores/metadata';
import type { DutyCycle } from '~/types/metadata';

const store = useMetadataStore();

const profiles = [
  {
    type: 'earthmoving' as const,
    icon: '🚜',
    name: 'Земляные работы',
    description: 'Экскаватор: копание, поворот, разгрузка',
    defaultDistribution: { digging: 40, swing: 30, unloading: 20, idle: 10 }
  },
  {
    type: 'loading' as const,
    icon: '🏗️',
    name: 'Погрузка',
    description: 'Погрузчик: подъём, движение, опускание',
    defaultDistribution: { lifting: 35, moving: 35, lowering: 20, idle: 10 }
  },
  {
    type: 'lifting' as const,
    icon: '🏗️',
    name: 'Подъём грузов',
    description: 'Кран: поднятие, опускание',
    defaultDistribution: { lifting: 50, lowering: 40, idle: 10 }
  },
  {
    type: 'custom' as const,
    icon: '⚙️',
    name: 'Другой',
    description: 'Пользовательский профиль',
    defaultDistribution: { operation_1: 50, operation_2: 30, operation_3: 20 }
  }
];

const formData = reactive<DutyCycle>({
  profile_type: store.wizardState.system.duty_cycle?.profile_type || '' as any,
  load_distribution: store.wizardState.system.duty_cycle?.load_distribution || {},
  peak_load_frequency: store.wizardState.system.duty_cycle?.peak_load_frequency || 'regular',
  break_interval_minutes: store.wizardState.system.duty_cycle?.break_interval_minutes || 30,
  ambient_conditions: store.wizardState.system.duty_cycle?.ambient_conditions || {
    temp_min: -20,
    temp_max: 40,
    dusty: false,
    humid: false,
    high_vibration: false,
    hot_environment: false,
    high_altitude: false
  }
});

const loadTotal = computed(() => {
  return Object.values(formData.load_distribution).reduce((sum, val) => sum + (val || 0), 0);
});

function selectProfile(type: DutyCycle['profile_type']) {
  formData.profile_type = type;
  const profile = profiles.find(p => p.type === type);
  if (profile) {
    // ✅ Убираем undefined ключи:
    formData.load_distribution = Object.fromEntries(
      Object.entries(profile.defaultDistribution).filter(([_, v]) => v !== undefined)
    ) as Record<string, number>;
  }
}

function getLoadLabel(key: string): string {
  const labels: Record<string, string> = {
    digging: 'Копание грунта',
    swing: 'Поворот с ковшом',
    unloading: 'Разгрузка',
    lifting: 'Подъём груза',
    moving: 'Горизонтальное движение',
    lowering: 'Опускание',
    idle: 'Холостой ход',
    operation_1: 'Операция 1',
    operation_2: 'Операция 2',
    operation_3: 'Операция 3'
  };
  return labels[key] || key;
}

function getPeakLoadDescription(freq: string): string {
  const descriptions: Record<string, string> = {
    rare: 'Низкий риск перегрева',
    regular: 'Средний риск перегрева',
    frequent: 'Высокий риск перегрева — требуется усиленное охлаждение',
    constant: 'Критический риск — необходима проверка системы охлаждения'
  };
  return descriptions[freq] || '';
}

watch(formData, () => {
  store.updateBasicInfo({ duty_cycle: { ...formData } });
}, { deep: true });
</script>

<style scoped>
.level-4 {
  padding: 1rem;
}

.form-sections {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.form-section {
  padding: 1.5rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
}

.section-title {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 1rem;
  color: #374151;
}

.profile-selector {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1rem;
}

.profile-card {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 0.5rem;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
  text-align: left;
}

.profile-card:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.profile-card.selected {
  border-color: #3b82f6;
  background: #dbeafe;
}

.profile-icon {
  font-size: 2rem;
}

.profile-info {
  flex: 1;
}

.profile-name {
  font-weight: 600;
  font-size: 0.875rem;
  color: #374151;
}

.profile-description {
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 0.25rem;
}

.load-distribution {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
}

.distribution-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.distribution-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}

.distribution-input {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.form-input {
  flex: 1;
  padding: 0.625rem 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
}

.distribution-unit {
  font-size: 0.875rem;
  color: #6b7280;
}

.total-indicator {
  margin-top: 1rem;
  padding: 0.75rem;
  background: #ecfdf5;
  border: 1px solid #a7f3d0;
  border-radius: 0.5rem;
  font-weight: 600;
  color: #065f46;
}

.total-indicator.error {
  background: #fef2f2;
  border-color: #fecaca;
  color: #991b1b;
}

.error-text {
  font-weight: normal;
}

.form-select {
  width: 100%;
  padding: 0.625rem 0.75rem;
  border: 1px solid #d1d5db;
  border-radius: 0.5rem;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}

.help-text {
  font-size: 0.75rem;
  color: #6b7280;
}

.slider-container {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.5rem;
}

.slider {
  flex: 1;
  height: 6px;
  border-radius: 3px;
  background: #e5e7eb;
  outline: none;
  cursor: pointer;
}

.slider::-webkit-slider-thumb {
  width: 20px;
  height: 20px;
  border-radius: 50%;
  background: #3b82f6;
  cursor: pointer;
}

.slider-value {
  font-weight: 600;
  font-size: 0.875rem;
  color: #374151;
  min-width: 80px;
}

.conditions-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
  margin-bottom: 1rem;
}

.condition-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.condition-label {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
}

.checkboxes {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.checkbox-item {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
}

.checkbox-item input[type="checkbox"] {
  width: 18px;
  height: 18px;
  cursor: pointer;
}

.checkbox-item span {
  font-size: 0.875rem;
  color: #374151;
}
</style>
