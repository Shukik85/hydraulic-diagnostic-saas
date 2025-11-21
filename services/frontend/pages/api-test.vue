<template>
  <div class="p-8 space-y-10">
    <h1 class="text-3xl font-bold text-blue-700">API Test Dashboard</h1>
    <section class="bg-white rounded-xl shadow p-6">
      <h2 class="text-xl font-semibold mb-2">System Status</h2>
      <div v-if="systemStatus.loading" class="text-gray-500">Loading...</div>
      <div v-else-if="systemStatus.error" class="text-red-600">{{ systemStatus.error?.error?.message || 'Unknown error' }}</div>
      <div v-else-if="systemStatus.data">
        <div class="text-lg font-bold">
          Health Score: <span class="text-green-600">{{ systemStatus.data.health_score }}</span>
        </div>
        <div v-for="comp in systemStatus.data.component_statuses" :key="comp.component_id" class="mt-2">
          <span class="font-medium">Component:</span> {{ comp.name }}
          <span :class="getComponentStatusColor(comp.status)">Status: {{ comp.status }}</span>
        </div>
      </div>
      <button @click="systemStatusRefresh" class="mt-4 bg-blue-600 text-white py-2 px-4 rounded">Manual Refresh</button>
    </section>

    <section class="bg-white rounded-xl shadow p-6">
      <h2 class="text-xl font-semibold mb-2">Anomalies List</h2>
      <div v-if="anomaliesState.loading" class="text-gray-500">Loading...</div>
      <div v-else-if="anomaliesState.error" class="text-red-600">{{ anomaliesState.error?.error?.message || 'Unknown error' }}</div>
      <div v-else-if="anomaliesState.data">
        <div>
          <label>Severity:</label>
          <select v-model="severityFilter" class="border p-1 rounded ml-2">
            <option value="">All</option>
            <option value="normal">normal</option>
            <option value="warning">warning</option>
            <option value="critical">critical</option>
          </select>
        </div>
        <div class="mt-2">
          <ul>
            <li v-for="anomaly in anomaliesState.data.items" :key="anomaly.prediction_id" :class="getSeverityColor(anomaly.severity)">
              <span class="font-bold">Score:</span> {{ anomaly.anomaly_score != null ? formatAnomalyScore(anomaly.anomaly_score) : 'N/A' }}
              <span class="ml-4 font-semibold">Severity:</span> {{ anomaly.severity }}
            </li>
          </ul>
        </div>
        <div class="mt-4">
          <button @click="prevPage" :disabled="anomaliesPage <= 1" class="bg-gray-200 py-1 px-3 rounded mr-2">Prev</button>
          <button @click="nextPage" :disabled="anomaliesPage >= (anomaliesState.data?.pagination?.pages || 1)" class="bg-gray-200 py-1 px-3 rounded">Next</button>
          <span class="ml-4">Page {{ anomaliesPage }} of {{ anomaliesState.data?.pagination?.pages || 1 }}</span>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useSystemStatus } from '../composables/useSystemStatus'
import { useAnomalies } from '../composables/useAnomalies'
import type { AnomalySeverity } from '../types/api'

const systemId = '550e8400-e29b-41d4-a716-446655440000' // demo UUID

const { state: systemStatus, load: systemStatusRefresh } = useSystemStatus(systemId, 10000)

const anomaliesPage = ref(1)
const severityFilter = ref<AnomalySeverity | undefined>()
const { state: anomaliesState, page, severity, load } = useAnomalies(systemId, { page: anomaliesPage.value, severity: severityFilter.value })

watch(anomaliesPage, (v) => { page.value = v; load() })
watch(severityFilter, (v) => { severity.value = v; load() })

function prevPage() { if (anomaliesPage.value > 1) anomaliesPage.value-- }
function nextPage() { anomaliesPage.value++ }

function formatAnomalyScore(score: number): string {
  return score.toFixed(3)
}

function getSeverityColor(severity: string): string {
  const colors: Record<string, string> = {
    normal: 'bg-green-50 text-green-800',
    warning: 'bg-yellow-50 text-yellow-800',
    critical: 'bg-red-50 text-red-800'
  }
  return colors[severity] || 'bg-gray-50 text-gray-800'
}

function getComponentStatusColor(status: string): string {
  const colors: Record<string, string> = {
    online: 'text-green-600',
    offline: 'text-red-600',
    warning: 'text-yellow-600',
    error: 'text-red-800'
  }
  return colors[status] || 'text-gray-600'
}
</script>

<style scoped>
.text-green-600 { color: #16a34a; }
.text-red-600 { color: #dc2626; }
.bg-red-50 { background-color: #fef2f2; }
.bg-green-50 { background-color: #f0fdf4; }
.bg-yellow-50 { background-color: #fefce8; }
</style>
