<!--
  DiagnosticsDashboard.vue — Главный dashboard для диагностики
  
  Features:
  - Grid of sensor charts
  - System architecture graph
  - GNN inference trigger
  - Anomaly detection results
  - Recommendations panel
  - Export to CSV/PDF
-->
<template>
  <div class="diagnostics-dashboard space-y-6">
    <!-- Header with actions -->
    <div class="flex items-center justify-between">
      <div>
        <h2 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
          Diagnostics Dashboard
        </h2>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Real-time monitoring and anomaly detection
        </p>
      </div>
      
      <div class="flex items-center gap-3">
        <!-- Run GNN Inference -->
        <UButton
          color="primary"
          icon="i-heroicons-cpu-chip"
          :loading="isRunningInference"
          @click="runGNNInference"
        >
          Run GNN Analysis
        </UButton>
        
        <!-- Export -->
        <UDropdown :items="exportItems">
          <UButton
            icon="i-heroicons-arrow-down-tray"
            variant="outline"
          >
            Export
          </UButton>
        </UDropdown>
      </div>
    </div>
    
    <!-- System Architecture Graph -->
    <GraphView
      :components="components"
      :adjacency-matrix="adjacencyMatrix"
      :anomaly-scores="anomalyScores"
      @component-select="onComponentSelect"
    />
    
    <!-- GNN Results -->
    <div v-if="gnnResults" class="grid grid-cols-3 gap-6">
      <!-- Overall Health Score -->
      <BaseCard>
        <div class="text-center">
          <p class="text-sm text-gray-500 dark:text-gray-400 mb-2">
            System Health Score
          </p>
          <div class="text-4xl font-bold" :class="getHealthScoreColor(gnnResults.health_score)">
            {{ (gnnResults.health_score * 100).toFixed(1) }}%
          </div>
          <p class="text-xs text-gray-500 mt-2">
            Last updated: {{ formatTime(gnnResults.timestamp) }}
          </p>
        </div>
      </BaseCard>
      
      <!-- Detected Anomalies -->
      <BaseCard>
        <div class="text-center">
          <p class="text-sm text-gray-500 dark:text-gray-400 mb-2">
            Detected Anomalies
          </p>
          <div class="text-4xl font-bold text-red-600 dark:text-red-400">
            {{ gnnResults.anomalies?.length || 0 }}
          </div>
          <p class="text-xs text-gray-500 mt-2">
            {{ criticalCount }} critical, {{ warningCount }} warning
          </p>
        </div>
      </BaseCard>
      
      <!-- Prediction Confidence -->
      <BaseCard>
        <div class="text-center">
          <p class="text-sm text-gray-500 dark:text-gray-400 mb-2">
            Prediction Confidence
          </p>
          <div class="text-4xl font-bold text-blue-600 dark:text-blue-400">
            {{ (gnnResults.confidence * 100).toFixed(1) }}%
          </div>
          <UProgress
            :value="gnnResults.confidence * 100"
            class="mt-3"
          />
        </div>
      </BaseCard>
    </div>
    
    <!-- Component Anomaly Scores -->
    <BaseCard v-if="gnnResults?.component_scores">
      <h3 class="text-lg font-semibold mb-4">
        Component Anomaly Scores
      </h3>
      
      <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div
          v-for="(score, componentId) in gnnResults.component_scores"
          :key="componentId"
          class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg"
        >
          <div class="flex items-center justify-between mb-2">
            <p class="text-sm font-medium text-gray-900 dark:text-gray-100">
              {{ getComponentName(componentId) }}
            </p>
            <StatusBadge :status="getStatusFromScore(score)" />
          </div>
          
          <div class="flex items-center gap-2">
            <UProgress
              :value="score * 100"
              :color="getProgressColor(score)"
              size="sm"
              class="flex-1"
            />
            <span class="text-xs font-semibold" :class="getScoreColor(score)">
              {{ (score * 100).toFixed(0) }}%
            </span>
          </div>
        </div>
      </div>
    </BaseCard>
    
    <!-- Recommendations -->
    <BaseCard v-if="gnnResults?.recommendations">
      <h3 class="text-lg font-semibold mb-4">
        Recommendations
      </h3>
      
      <div class="space-y-3">
        <UAlert
          v-for="(rec, index) in gnnResults.recommendations"
          :key="index"
          :color="getRecommendationColor(rec.priority)"
          :icon="getRecommendationIcon(rec.priority)"
          :title="rec.title"
          :description="rec.description"
        >
          <template v-if="rec.actions" #actions>
            <UButton
              v-for="action in rec.actions"
              :key="action"
              size="xs"
              variant="ghost"
            >
              {{ action }}
            </UButton>
          </template>
        </UAlert>
      </div>
    </BaseCard>
    
    <!-- Sensor Charts Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <SensorChart
        v-for="sensor in sensors"
        :key="sensor.id"
        :sensor-id="sensor.id"
        :sensor-name="sensor.name"
        :sensor-type="sensor.type"
        :unit="sensor.unit"
        :expected-range="sensor.expectedRange"
        @anomaly-click="onAnomalyClick"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { ComponentMetadata } from '~/types/metadata'

interface Props {
  equipmentId: string
  components: ComponentMetadata[]
  adjacencyMatrix: number[][]
  sensors: Array<{
    id: string
    name: string
    type: string
    unit: string
    expectedRange?: { min: number; max: number }
  }>
}

const props = defineProps<Props>()

const api = useApiAdvanced()
const toast = useToast()

const isRunningInference = ref(false)
const gnnResults = ref<any>(null)
const anomalyScores = ref<Record<string, number>>({})

// Computed
const criticalCount = computed(() => {
  return gnnResults.value?.anomalies?.filter((a: any) => a.severity === 'critical').length || 0
})

const warningCount = computed(() => {
  return gnnResults.value?.anomalies?.filter((a: any) => a.severity === 'warning').length || 0
})

// Export items
const exportItems = [
  [
    {
      label: 'Export to CSV',
      icon: 'i-heroicons-document-text',
      click: () => exportToCSV()
    },
    {
      label: 'Export to PDF',
      icon: 'i-heroicons-document',
      click: () => exportToPDF()
    }
  ]
]

// Run GNN inference
async function runGNNInference() {
  isRunningInference.value = true
  
  try {
    const response = await api.post<any>(
      '/api/gnn/infer',
      {
        equipment_id: props.equipmentId,
        components: props.components,
        adjacency_matrix: props.adjacencyMatrix
      }
    )
    
    gnnResults.value = response
    anomalyScores.value = response.component_scores || {}
    
    toast.add({
      title: 'GNN Analysis Complete',
      description: `Detected ${response.anomalies?.length || 0} anomalies`,
      color: response.anomalies?.length > 0 ? 'yellow' : 'green'
    })
  } catch (error: any) {
    toast.add({
      title: 'GNN Analysis Failed',
      description: error.message,
      color: 'red'
    })
  } finally {
    isRunningInference.value = false
  }
}

// Export functions
function exportToCSV() {
  if (!gnnResults.value) {
    toast.add({
      title: 'No data to export',
      description: 'Run GNN analysis first',
      color: 'yellow'
    })
    return
  }
  
  // Generate CSV
  const csvData = [
    ['Component', 'Anomaly Score', 'Status'],
    ...Object.entries(gnnResults.value.component_scores || {}).map(([id, score]: [string, any]) => [
      getComponentName(id),
      (score * 100).toFixed(2) + '%',
      getStatusFromScore(score)
    ])
  ]
  
  const csv = csvData.map(row => row.join(',')).join('\n')
  const blob = new Blob([csv], { type: 'text/csv' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `diagnostics_${props.equipmentId}_${Date.now()}.csv`
  a.click()
  URL.revokeObjectURL(url)
  
  toast.add({
    title: 'Export successful',
    description: 'CSV file downloaded',
    color: 'green'
  })
}

function exportToPDF() {
  toast.add({
    title: 'PDF export',
    description: 'Feature coming soon',
    color: 'blue'
  })
}

// Helper functions
function getComponentName(componentId: string): string {
  const comp = props.components.find(c => c.id === componentId)
  return comp?.name || componentId
}

function getHealthScoreColor(score: number): string {
  if (score >= 0.8) return 'text-green-600 dark:text-green-400'
  if (score >= 0.6) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

function getStatusFromScore(score: number): string {
  if (score < 0.3) return 'operational'
  if (score < 0.7) return 'warning'
  return 'critical'
}

function getScoreColor(score: number): string {
  if (score < 0.3) return 'text-green-600 dark:text-green-400'
  if (score < 0.7) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

function getProgressColor(score: number): string {
  if (score < 0.3) return 'green'
  if (score < 0.7) return 'yellow'
  return 'red'
}

function getRecommendationColor(priority: string): string {
  const colors: Record<string, string> = {
    high: 'red',
    medium: 'yellow',
    low: 'blue'
  }
  return colors[priority] || 'gray'
}

function getRecommendationIcon(priority: string): string {
  const icons: Record<string, string> = {
    high: 'i-heroicons-exclamation-triangle',
    medium: 'i-heroicons-exclamation-circle',
    low: 'i-heroicons-information-circle'
  }
  return icons[priority] || 'i-heroicons-information-circle'
}

function formatTime(timestamp: string | number): string {
  return new Date(timestamp).toLocaleString()
}

function onComponentSelect(component: ComponentMetadata) {
  console.log('Component selected:', component)
}

function onAnomalyClick(data: any) {
  console.log('Anomaly clicked:', data)
}
</script>
