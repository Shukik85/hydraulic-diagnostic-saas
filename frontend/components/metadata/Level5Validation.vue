<!-- components/metadata/Level5Validation.vue -->
<template>
  <div class="level-5">
    <h2 class="text-xl font-semibold mb-4">5. Финальная валидация и отправка</h2>

    <p class="text-gray-600 mb-6">
      Проверка полноты данных и готовности к обучению GNN модели.
    </p>

    <!-- Overall Progress -->
    <div class="overall-progress mb-6">
      <h3 class="text-lg font-semibold mb-3">Общая готовность системы</h3>
      <div class="progress-circle" :class="progressClass">
        <svg width="120" height="120">
          <circle cx="60" cy="60" r="54" fill="none" stroke="#e5e7eb" stroke-width="8" />
          <circle cx="60" cy="60" r="54" fill="none" :stroke="progressColor" stroke-width="8" stroke-linecap="round"
            :stroke-dasharray="circumference" :stroke-dashoffset="dashOffset" transform="rotate(-90 60 60)" />
        </svg>
        <div class="progress-text">
          <span class="progress-pct">{{ store.completeness }}%</span>
          <span class="progress-label">готово</span>
        </div>
      </div>
      <p class="text-center mt-4 text-sm">
        <span v-if="store.completeness < 50" class="text-red-600">
          ❌ Недостаточно данных для обучения модели
        </span>
        <span v-else-if="store.completeness < 70" class="text-amber-600">
          ⚠ Хорошо, но рекомендуется заполнить больше полей
        </span>
        <span v-else class="text-green-600">
          ✅ Отлично! Готово к обучению GNN модели
        </span>
      </p>
    </div>

    <!-- Validation Errors -->
    <div v-if="validationErrors.length > 0" class="validation-section error-section mb-6">
      <h3 class="section-title">❌ Ошибки валидации ({{ validationErrors.length }})</h3>
      <ul class="validation-list">
        <li v-for="(error, i) in validationErrors" :key="i" class="validation-item error">
          <span class="item-icon">⚠</span>
          <div class="item-content">
            <div class="item-message">{{ error.error }}</div>
            <div v-if="error.suggestion" class="item-suggestion">{{ error.suggestion }}</div>
          </div>
        </li>
      </ul>
    </div>

    <!-- Missing Critical Fields -->
    <div v-if="incompleteness.critical_missing.length > 0" class="validation-section warning-section mb-6">
      <h3 class="section-title">⚠ Критичные поля ({{ incompleteness.critical_missing.length }})</h3>
      <p class="section-description">Эти поля обязательны для корректной работы модели</p>
      <ul class="validation-list">
        <li v-for="field in incompleteness.critical_missing" :key="field" class="validation-item warning">
          <span class="item-icon">🔴</span>
          <div class="item-content">{{ field }}</div>
        </li>
      </ul>
    </div>

    <!-- Missing Secondary Fields -->
    <div v-if="incompleteness.secondary_missing.length > 0" class="validation-section info-section mb-6">
      <h3 class="section-title">ℹ Вспомогательные поля ({{ incompleteness.secondary_missing.length }})</h3>
      <p class="section-description">Необязательные, но улучшат точность модели</p>
      <ul class="validation-list">
        <li v-for="field in incompleteness.secondary_missing" :key="field" class="validation-item info">
          <span class="item-icon">🟡</span>
          <div class="item-content">{{ field }}</div>
        </li>
      </ul>
    </div>

    <!-- Inferred Values -->
    <div v-if="Object.keys(incompleteness.inferred_values).length > 0" class="validation-section inferred-section mb-6">
      <h3 class="section-title">🤖 Инферированные значения ({{ Object.keys(incompleteness.inferred_values).length }})
      </h3>
      <p class="section-description">Автоматически заполнены на основе других данных</p>
      <ul class="validation-list">
        <li v-for="(data, field) in incompleteness.inferred_values" :key="field" class="validation-item inferred">
          <span class="item-icon">💡</span>
          <div class="item-content">
            <div class="item-message">{{ field }}: {{ JSON.stringify(data.value) }}</div>
            <div class="item-meta">
              <span class="meta-method">{{ data.method }}</span>
              <span class="meta-confidence" :class="getConfidenceClass(data.confidence)">
                Confidence: {{ (data.confidence * 100).toFixed(0) }}%
              </span>
            </div>
          </div>
        </li>
      </ul>
    </div>

    <!-- Summary Stats -->
    <div class="summary-stats mb-6">
      <div class="stat-card">
        <div class="stat-value">{{ store.componentsCount }}</div>
        <div class="stat-label">Компонентов</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ adjacencyEdgesCount }}</div>
        <div class="stat-label">Связей</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ store.completeness }}%</div>
        <div class="stat-label">Заполнено</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ confidenceScore.toFixed(2) }}</div>
        <div class="stat-label">Уверенность</div>
      </div>
    </div>

    <!-- Actions -->
    <div class="actions">
      <button @click="runInference" class="btn btn-secondary" :disabled="isSubmitting">
        Инферировать недостающие значения
      </button>

      <button @click="submitMetadata" class="btn btn-primary" :disabled="isSubmitting || !canSubmit">
        <span v-if="isSubmitting">Отправка...</span>
        <span v-else>{{ store.completeness >= 70 ? 'Завершить настройку' : 'Сохранить с пробелами' }}</span>
      </button>
    </div>

    <!-- Result Modal -->
    <div v-if="showResultModal" class="modal-overlay" @click="closeModal">
      <div class="modal" @click.stop>
        <div v-if="submitSuccess" class="modal-content success">
          <div class="modal-icon">✅</div>
          <h3 class="modal-title">Метаданные успешно сохранены!</h3>
          <p class="modal-description">
            Система готова к обучению GNN модели.<br>
            Полнота данных: {{ store.completeness }}%
          </p>
          <button @click="goToDashboard" class="btn btn-primary">Перейти в Dashboard</button>
        </div>

        <div v-else class="modal-content error">
          <div class="modal-icon">❌</div>
          <h3 class="modal-title">Ошибка отправки</h3>
          <p class="modal-description">{{ submitError }}</p>
          <button @click="closeModal" class="btn btn-secondary">Закрыть</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useMetadataStore } from '~/stores/metadata';

const store = useMetadataStore();
const router = useRouter();

const isSubmitting = ref(false);
const showResultModal = ref(false);
const submitSuccess = ref(false);
const submitError = ref('');

const validationErrors = computed(() => store.validateConsistency());

const incompleteness = computed(() => store.wizardState.incompleteness_report);

const adjacencyEdgesCount = computed(() => {
  const matrix = store.wizardState.system.adjacency_matrix || [];
  return matrix.reduce((sum, row) => sum + row.reduce((s, v) => s + v, 0), 0);
});

const confidenceScore = computed(() => {
  const inferred = Object.values(incompleteness.value.inferred_values);
  if (inferred.length === 0) return 1.0;

  const avgConfidence = inferred.reduce((sum, v) => sum + (v as any).confidence, 0) / inferred.length;
  return avgConfidence;
});

const canSubmit = computed(() => {
  return validationErrors.value.length === 0 && store.componentsCount > 0;
});

// Progress Circle
const circumference = 2 * Math.PI * 54;
const dashOffset = computed(() => {
  return circumference - (store.completeness / 100) * circumference;
});

const progressColor = computed(() => {
  if (store.completeness < 50) return '#ef4444';
  if (store.completeness < 70) return '#f59e0b';
  return '#10b981';
});

const progressClass = computed(() => {
  if (store.completeness < 50) return 'progress-low';
  if (store.completeness < 70) return 'progress-medium';
  return 'progress-high';
});

function getConfidenceClass(confidence: number): string {
  if (confidence < 0.5) return 'confidence-low';
  if (confidence < 0.7) return 'confidence-medium';
  return 'confidence-high';
}

function runInference() {
  store.inferMissingValues();
}

async function submitMetadata() {
  isSubmitting.value = true;

  try {
    const result = await store.submitMetadata();

    if (result.success) {
      submitSuccess.value = true;
      showResultModal.value = true;
    } else {
      submitSuccess.value = false;
      const error = result.error as any;
      submitError.value = error?.message || 'Неизвестная ошибка';
      showResultModal.value = true;
    }
  } catch (error: any) {
    submitSuccess.value = false;
    submitError.value = error.message || 'Ошибка сети';
    showResultModal.value = true;
  } finally {
    isSubmitting.value = false;
  }
}

function closeModal() {
  showResultModal.value = false;
}

function goToDashboard() {
  router.push('/dashboard');
}
</script>

<style scoped>
.level-5 {
  padding: 1rem;
}

.overall-progress {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 2rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
}

.progress-circle {
  position: relative;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.progress-pct {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  color: #374151;
}

.progress-label {
  font-size: 0.75rem;
  color: #6b7280;
}

.validation-section {
  padding: 1.5rem;
  border-radius: 0.75rem;
}

.error-section {
  background: #fef2f2;
  border: 1px solid #fecaca;
}

.warning-section {
  background: #fffbeb;
  border: 1px solid #fde68a;
}

.info-section {
  background: #eff6ff;
  border: 1px solid #bfdbfe;
}

.inferred-section {
  background: #f0fdf4;
  border: 1px solid #bbf7d0;
}

.section-title {
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
  color: #374151;
}

.section-description {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 1rem;
}

.validation-list {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.validation-item {
  display: flex;
  gap: 0.75rem;
  align-items: start;
}

.item-icon {
  font-size: 1.25rem;
}

.item-content {
  flex: 1;
}

.item-message {
  font-size: 0.875rem;
  color: #374151;
  font-weight: 500;
}

.item-suggestion {
  font-size: 0.75rem;
  color: #6b7280;
  margin-top: 0.25rem;
}

.item-meta {
  display: flex;
  gap: 1rem;
  margin-top: 0.5rem;
}

.meta-method {
  font-size: 0.75rem;
  color: #6b7280;
  font-style: italic;
}

.meta-confidence {
  font-size: 0.75rem;
  font-weight: 600;
}

.confidence-low {
  color: #ef4444;
}

.confidence-medium {
  color: #f59e0b;
}

.confidence-high {
  color: #10b981;
}

.summary-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
}

.stat-card {
  padding: 1.5rem;
  background: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.75rem;
  text-align: center;
}

.stat-value {
  font-size: 2rem;
  font-weight: 700;
  color: #3b82f6;
}

.stat-label {
  font-size: 0.875rem;
  color: #6b7280;
  margin-top: 0.5rem;
}

.actions {
  display: flex;
  gap: 1rem;
  justify-content: flex-end;
}

.btn {
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 600;
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

.modal-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 50;
}

.modal {
  background: white;
  border-radius: 0.75rem;
  padding: 2rem;
  max-width: 400px;
  width: 90%;
}

.modal-content {
  text-align: center;
}

.modal-icon {
  font-size: 4rem;
  margin-bottom: 1rem;
}

.modal-title {
  font-size: 1.25rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
}

.modal-description {
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 1.5rem;
}
</style>
