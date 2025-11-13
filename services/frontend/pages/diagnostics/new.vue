<script setup lang="ts">
/**
 * New Diagnosis Page - Type-safe with Generated API
 * 
 * Run new diagnosis with:
 * - System selection
 * - Real-time progress
 * - GNN + RAG integration
 * - Type-safe results
 */

import { useGeneratedApi } from '~/composables/useGeneratedApi'
import type { System, DiagnosisRequest, DiagnosisResult, RAGInterpretation } from '~/generated/api'

definePageMeta({
  middleware: ['auth'],
  layout: 'dashboard'
})

// Composables
const api = useGeneratedApi()
const { success, error: notifyError } = useNotifications()

// State
const systems = ref<System[]>([])
const selectedSystemId = ref<string>('')
const running = ref(false)
const progress = ref(0)
const currentStage = ref<'idle' | 'gnn' | 'rag' | 'complete'>('idle')

// Results
const gnnResult = ref<DiagnosisResult | null>(null)
const ragInterpretation = ref<RAGInterpretation | null>(null)

// Load systems
async function loadSystems() {
  try {
    systems.value = await api.equipment.getSystems()
  } catch (err) {
    notifyError('Ошибка загрузки систем')
  }
}

// Run diagnosis
async function runDiagnosis() {
  if (!selectedSystemId.value) {
    notifyError('Выберите систему')
    return
  }
  
  running.value = true
  progress.value = 0
  currentStage.value = 'gnn'
  
  try {
    // 1. Get system data
    progress.value = 10
    const system = await api.equipment.getSystem(selectedSystemId.value)
    
    // 2. Prepare sensor readings
    progress.value = 20
    const request: DiagnosisRequest = {
      system_id: selectedSystemId.value,
      sensor_readings: [], // TODO: Get actual sensor readings
      time_window: 3600,
      metadata: {
        equipment_type: system.equipment_type,
        manufacturer: system.manufacturer,
        model: system.model
      }
    }
    
    // 3. Run GNN diagnosis
    progress.value = 30
    currentStage.value = 'gnn'
    gnnResult.value = await api.gnn.runDiagnosis(request)
    progress.value = 70
    
    // 4. Get RAG interpretation
    currentStage.value = 'rag'
    ragInterpretation.value = await api.rag.interpretDiagnosis({
      gnnResult: gnnResult.value,
      equipmentContext: {
        equipment_type: system.equipment_type,
        manufacturer: system.manufacturer
      }
    })
    progress.value = 100
    
    currentStage.value = 'complete'
    success('Диагностика завершена')
  } catch (err) {
    notifyError('Ошибка диагностики')
    console.error(err)
    currentStage.value = 'idle'
  } finally {
    running.value = false
  }
}

onMounted(() => loadSystems())
</script>

<template>
  <div class="diagnosis-page">
    <h1 class="text-3xl font-bold mb-8">Новая диагностика</h1>
    
    <!-- System selection -->
    <div class="form-section mb-6">
      <label class="form-label">Выберите систему</label>
      <select v-model="selectedSystemId" class="form-input">
        <option value="">Выберите...</option>
        <option v-for="system in systems" :key="system.id" :value="system.id">
          {{ system.name }} ({{ system.status }})
        </option>
      </select>
    </div>
    
    <!-- Run button -->
    <button
      :disabled="!selectedSystemId || running"
      @click="runDiagnosis"
      class="btn-primary mb-6"
    >
      {{ running ? 'Выполнение...' : 'Запустить диагностику' }}
    </button>
    
    <!-- Progress -->
    <div v-if="running" class="progress-section mb-6">
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: `${progress}%` }" />
      </div>
      <p class="text-sm text-gray-600 mt-2">
        {{ currentStage === 'gnn' ? 'GNN анализ...' : 'RAG интерпретация...' }}
      </p>
    </div>
    
    <!-- Results -->
    <div v-if="gnnResult" class="results-section">
      <h2 class="text-2xl font-bold mb-4">Результаты</h2>
      
      <!-- GNN Results -->
      <div class="result-card mb-6">
        <h3 class="text-lg font-semibold mb-3">GNN Анализ</h3>
        <div class="grid grid-cols-2 gap-4">
          <div>
            <p class="text-sm text-gray-500">Оценка аномалии</p>
            <p class="text-2xl font-bold">{{ (gnnResult.anomaly_score * 100).toFixed(1) }}%</p>
          </div>
          <div>
            <p class="text-sm text-gray-500">Обнаружено аномалий</p>
            <p class="text-2xl font-bold">{{ gnnResult.anomalies?.length || 0 }}</p>
          </div>
        </div>
      </div>
      
      <!-- RAG Interpretation -->
      <div v-if="ragInterpretation" class="result-card">
        <RAGInterpretation :interpretation="ragInterpretation" />
      </div>
    </div>
  </div>
</template>

<style scoped>
.diagnosis-page {
  @apply container mx-auto px-4 py-8;
}

.form-section {
  @apply bg-white dark:bg-gray-800 rounded-lg p-6 border;
}

.form-label {
  @apply block text-sm font-medium mb-2;
}

.form-input {
  @apply w-full px-4 py-2 border rounded-lg;
}

.btn-primary {
  @apply px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50;
}

.progress-section {
  @apply bg-white dark:bg-gray-800 rounded-lg p-6 border;
}

.progress-bar {
  @apply w-full h-2 bg-gray-200 rounded-full overflow-hidden;
}

.progress-fill {
  @apply h-full bg-blue-600 transition-all duration-300;
}

.results-section {
  @apply space-y-6;
}

.result-card {
  @apply bg-white dark:bg-gray-800 rounded-lg p-6 border;
}
</style>
