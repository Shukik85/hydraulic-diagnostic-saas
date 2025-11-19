<!-- components/metadata/WizardLayout.vue -->
<template>
  <div class="metadata-wizard">
    <!-- Progress Header -->
    <div class="wizard-header">
      <h1 class="text-2xl font-bold mb-4">Настройка метаданных гидросистемы</h1>

      <!-- Progress Bar -->
      <div class="progress-container mb-6">
        <div class="progress-bar">
          <div class="progress-fill" :style="{ width: `${store.completeness}%` }"></div>
        </div>
        <span class="progress-text">{{ store.completeness }}% заполнено</span>
      </div>

      <!-- Level Navigation -->
      <div class="level-nav">
        <button v-for="level in 5" :key="level" @click="goToLevel(level)" :class="[
          'level-btn',
          { 'active': currentLevel === level },
          { 'completed': store.wizardState.completed_levels.includes(level) }
        ]">
          <span class="level-number">{{ level }}</span>
          <span class="level-name">{{ levelNames[level - 1] }}</span>
        </button>
      </div>
    </div>

    <!-- Content Area -->
    <div class="wizard-content">
      <transition name="fade" mode="out-in">
        <component :is="currentLevelComponent" :key="currentLevel" />
      </transition>
    </div>

    <!-- Navigation Footer -->
    <div class="wizard-footer">
      <button @click="previousLevel" :disabled="currentLevel === 1" class="btn btn-secondary">
        Назад
      </button>

      <div class="flex items-center gap-4">
        <span v-if="!store.currentLevelValid" class="text-amber-600">
          ⚠ Заполните обязательные поля
        </span>

        <button v-if="currentLevel < 5" @click="nextLevel" :disabled="!store.currentLevelValid" class="btn btn-primary">
          Далее
        </button>

        <button v-else @click="submitMetadata" class="btn btn-success">
          Завершить настройку
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, computed } from 'vue'

import { useMetadataStore } from '~/stores/metadata';
import Level1BasicInfo from '~/components/metadata/Level1BasicInfo.vue';
import Level2GraphBuilder from '~/components/metadata/Level2GraphBuilder.vue';
import Level3ComponentForms from '~/components/metadata/Level3ComponentForms.vue';
import Level4DutyCycle from '~/components/metadata/Level4DutyCycle.vue';
import Level5Validation from '~/components/metadata/Level5Validation.vue';

const store = useMetadataStore();

const levelNames = [
  'Базовая информация',
  'Архитектура системы',
  'Характеристики компонентов',
  'Профиль нагрузки',
  'Валидация'
];

const currentLevel = computed(() => store.wizardState.current_level);

const currentLevelComponent = computed(() => {
  const components = {
    1: Level1BasicInfo,
    2: Level2GraphBuilder,
    3: Level3ComponentForms, // ✅ Контейнер, не отдельные формы
    4: Level4DutyCycle,
    5: Level5Validation
  };
  return components[currentLevel.value as keyof typeof components];
});

function goToLevel(level: number) {
  store.goToLevel(level);
}

function nextLevel() {
  if (store.currentLevelValid) {
    store.completeLevel(currentLevel.value);
    store.goToLevel(currentLevel.value + 1);
  }
}

function previousLevel() {
  if (currentLevel.value > 1) {
    store.goToLevel(currentLevel.value - 1);
  }
}

async function submitMetadata() {
  const result = await store.submitMetadata();
  if (result.success) {
    // Redirect или показать success message
    navigateTo('/dashboard');
  } else {
    // Показать ошибку
    alert('Ошибка отправки метаданных');
  }
}

// Load saved state on mount
onMounted(() => {
  store.loadFromLocalStorage();
});
</script>

<style scoped>
.metadata-wizard {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
}

.wizard-header {
  margin-bottom: 2rem;
}

.progress-container {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.progress-bar {
  flex: 1;
  height: 8px;
  background: #e5e7eb;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #10b981 0%, #059669 100%);
  transition: width 0.3s ease;
}

.progress-text {
  font-size: 0.875rem;
  color: #6b7280;
  min-width: 80px;
}

.level-nav {
  display: flex;
  gap: 0.5rem;
  overflow-x: auto;
  padding-bottom: 0.5rem;
}

.level-btn {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
  padding: 0.75rem 1rem;
  border: 2px solid #e5e7eb;
  border-radius: 0.5rem;
  background: white;
  cursor: pointer;
  transition: all 0.2s;
  min-width: 120px;
}

.level-btn:hover {
  border-color: #3b82f6;
  background: #eff6ff;
}

.level-btn.active {
  border-color: #3b82f6;
  background: #3b82f6;
  color: white;
}

.level-btn.completed {
  border-color: #10b981;
  background: #ecfdf5;
}

.level-btn.completed .level-number {
  background: #10b981;
  color: white;
}

.level-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #e5e7eb;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  font-size: 0.875rem;
}

.level-btn.active .level-number {
  background: white;
  color: #3b82f6;
}

.level-name {
  font-size: 0.75rem;
  text-align: center;
}

.wizard-content {
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  padding: 2rem;
  min-height: 500px;
  margin-bottom: 2rem;
}

.wizard-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  border-top: 1px solid #e5e7eb;
}

.btn {
  padding: 0.625rem 1.25rem;
  border-radius: 0.5rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.btn-secondary {
  background: #f3f4f6;
  color: #374151;
  border: 1px solid #d1d5db;
}

.btn-secondary:hover:not(:disabled) {
  background: #e5e7eb;
}

.btn-primary {
  background: #3b82f6;
  color: white;
  border: none;
}

.btn-primary:hover:not(:disabled) {
  background: #2563eb;
}

.btn-success {
  background: #10b981;
  color: white;
  border: none;
}

.btn-success:hover {
  background: #059669;
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
