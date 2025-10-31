<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div>
        <h1 class="u-h2">{{ t('diagnostics.title') }}</h1>
        <p class="u-body text-gray-600 mt-1">
          {{ t('diagnostics.subtitle') }}
        </p>
      </div>
      <button @click="showRunModal = true" class="u-btn u-btn-primary u-btn-md w-full sm:w-auto">
        <Icon name="heroicons:play" class="w-4 h-4 mr-2" />
        {{ t('diagnostics.runNew') }}
      </button>
    </div>

    <!-- KPI Overview -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('diagnostics.kpi.activeSessions') }}</h3>
          <div class="u-metric-icon bg-blue-100">
            <Icon name="heroicons:play-circle" class="w-5 h-5 text-blue-600" />
          </div>
        </div>
        <div class="u-metric-value">{{ activeSessions.length }}</div>
        <div class="u-metric-change text-gray-600 mt-2">
          <Icon name="heroicons:clock" class="w-4 h-4" />
          <span>{{ t('diagnostics.kpi.runningNow') }}</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('diagnostics.kpi.successRate') }}</h3>
          <div class="u-metric-icon bg-green-100">
            <Icon name="heroicons:check-badge" class="w-5 h-5 text-green-600" />
          </div>
        </div>
        <div class="u-metric-value">98.5%</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>{{ t('diagnostics.kpi.thisWeek', ['+2.1%']) }}</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('diagnostics.kpi.avgDuration') }}</h3>
          <div class="u-metric-icon bg-purple-100">
            <Icon name="heroicons:clock" class="w-5 h-5 text-purple-600" />
          </div>
        </div>
        <div class="u-metric-value">4.2<span class="text-lg text-gray-500">{{ t('ui.minutes') }}</span></div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-down" class="w-4 h-4" />
          <span>{{ t('diagnostics.kpi.faster', ['-0.8']) }}</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('diagnostics.kpi.issuesFound') }}</h3>
          <div class="u-metric-icon bg-red-100">
            <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-red-600" />
          </div>
        </div>
        <div class="u-metric-value">7</div>
        <div class="u-metric-change u-metric-change-negative mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>{{ t('diagnostics.kpi.needsAttention') }}</span>
        </div>
      </div>
    </div>

    <!-- Active Sessions -->
    <div v-if="activeSessions.length > 0">
      <h2 class="u-h4 mb-6">{{ t('diagnostics.activeSessions') }}</h2>
      <div class="space-y-4">
        <div v-for="session in activeSessions" :key="session.id" class="u-card p-4 sm:p-6">
          <div class="flex flex-col sm:flex-row sm:items-center gap-4">
            <div class="flex items-center gap-4 flex-1 min-w-0">
              <div class="w-3 h-3 bg-blue-500 rounded-full animate-pulse flex-shrink-0"></div>
              <div class="min-w-0">
                <p class="font-medium text-gray-900 truncate">{{ session.name }}</p>
                <p class="u-body-sm text-gray-500 truncate">
                  {{ session.equipment }} • {{ t('diagnostics.started') }} {{ session.startedAt }}
                </p>
              </div>
            </div>
            <div class="flex items-center gap-4 justify-between sm:justify-end">
              <div class="w-24 sm:w-32">
                <div class="flex items-center gap-2">
                  <div class="flex-1 bg-gray-200 rounded-full h-2">
                    <div
                      class="bg-blue-500 h-2 rounded-full u-transition-fast"
                      :style="{ width: session.progress + '%' }"
                    ></div>
                  </div>
                  <span class="u-body-sm font-medium text-gray-700 text-xs sm:text-sm">
                    {{ Math.round(session.progress) }}%
                  </span>
                </div>
              </div>
              <button @click="cancelSession(session.id)" class="u-btn u-btn-ghost u-btn-sm shrink-0">
                <Icon name="heroicons:x-mark" class="w-4 h-4 mr-1" />
                {{ t('ui.cancel') }}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Recent Results -->
    <div class="u-card">
      <div class="p-4 sm:p-6 border-b border-gray-200">
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h3 class="u-h4">{{ t('diagnostics.recentResults.title') }}</h3>
            <p class="u-body text-gray-600 mt-1">
              {{ t('diagnostics.recentResults.subtitle') }}
            </p>
          </div>
          <div class="flex items-center gap-2">
            <select class="u-input text-sm py-2 px-3 w-full sm:w-40">
              <option>{{ t('diagnostics.filters.allEquipment') }}</option>
              <option>HYD-001</option>
              <option>HYD-002</option>
              <option>HYD-003</option>
            </select>
            <button class="u-btn u-btn-secondary u-btn-sm flex-shrink-0">
              <Icon name="heroicons:funnel" class="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
      
      <div class="p-0 sm:p-6">
        <!-- Mobile: Card Layout -->
        <div class="sm:hidden space-y-4 p-4">
          <div v-for="result in recentResults" :key="result.id" class="u-card p-4">
            <div class="flex items-start justify-between gap-3 mb-3">
              <div class="flex-1 min-w-0">
                <h4 class="font-medium text-gray-900 truncate">{{ result.name }}</h4>
                <p class="u-body-sm text-gray-500">{{ result.equipment }}</p>
              </div>
              <span 
                class="u-badge flex-shrink-0"
                :class="getStatusBadgeClass(result.status)"
              >
                <Icon :name="getStatusIcon(result.status)" class="w-3 h-3" />
                {{ t(`diagnostics.status.${result.status}`) }}
              </span>
            </div>
            
            <div class="grid grid-cols-2 gap-4 mb-4">
              <div>
                <p class="text-xs text-gray-500 mb-1">{{ t('diagnostics.healthScore') }}</p>
                <div class="flex items-center gap-2">
                  <div class="w-8 h-2 bg-gray-200 rounded-full">
                    <div 
                      class="h-2 rounded-full"
                      :class="result.score >= 90 ? 'bg-green-500' : result.score >= 70 ? 'bg-yellow-500' : 'bg-red-500'"
                      :style="{ width: result.score + '%' }"
                    ></div>
                  </div>
                  <span class="text-sm font-medium">{{ result.score }}/100</span>
                </div>
              </div>
              <div>
                <p class="text-xs text-gray-500 mb-1">{{ t('diagnostics.issues') }}</p>
                <span 
                  class="u-badge text-xs"
                  :class="result.issuesFound === 0 ? 'u-badge-success' : result.issuesFound <= 2 ? 'u-badge-warning' : 'u-badge-error'"
                >
                  {{ result.issuesFound }} {{ t('diagnostics.issuesCount') }}
                </span>
              </div>
            </div>
            
            <div class="flex items-center justify-between">
              <span class="u-body-sm text-gray-500">{{ result.completedAt }}</span>
              <div class="flex items-center gap-2">
                <button @click="viewResult(result.id)" class="u-btn u-btn-ghost u-btn-sm">
                  <Icon name="heroicons:eye" class="w-4 h-4" />
                </button>
                <button class="u-btn u-btn-ghost u-btn-sm">
                  <Icon name="heroicons:arrow-down-tray" class="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Desktop: Table Layout -->
        <div class="hidden sm:block overflow-x-auto">
          <table class="u-table">
            <thead>
              <tr>
                <th>{{ t('diagnostics.table.name') }}</th>
                <th>{{ t('diagnostics.table.equipment') }}</th>
                <th>{{ t('diagnostics.table.healthScore') }}</th>
                <th>{{ t('diagnostics.table.issues') }}</th>
                <th>{{ t('diagnostics.table.completed') }}</th>
                <th>{{ t('diagnostics.table.status') }}</th>
                <th>{{ t('diagnostics.table.actions') }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="result in recentResults" :key="result.id">
                <td class="font-medium">{{ result.name }}</td>
                <td>{{ result.equipment }}</td>
                <td>
                  <div class="flex items-center gap-2">
                    <div class="w-12 h-2 bg-gray-200 rounded-full">
                      <div 
                        class="h-2 rounded-full"
                        :class="result.score >= 90 ? 'bg-green-500' : result.score >= 70 ? 'bg-yellow-500' : 'bg-red-500'"
                        :style="{ width: result.score + '%' }"
                      ></div>
                    </div>
                    <span class="u-body-sm font-medium">{{ result.score }}/100</span>
                  </div>
                </td>
                <td>
                  <span 
                    class="u-badge"
                    :class="result.issuesFound === 0 ? 'u-badge-success' : result.issuesFound <= 2 ? 'u-badge-warning' : 'u-badge-error'"
                  >
                    {{ result.issuesFound }} {{ t('diagnostics.issuesCount') }}
                  </span>
                </td>
                <td class="u-body-sm text-gray-500">{{ result.completedAt }}</td>
                <td>
                  <span 
                    class="u-badge"
                    :class="getStatusBadgeClass(result.status)"
                  >
                    <Icon :name="getStatusIcon(result.status)" class="w-3 h-3" />
                    {{ t(`diagnostics.status.${result.status}`) }}
                  </span>
                </td>
                <td>
                  <div class="flex items-center gap-2">
                    <button @click="viewResult(result.id)" class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:eye" class="w-4 h-4" />
                    </button>
                    <button class="u-btn u-btn-ghost u-btn-sm">
                      <Icon name="heroicons:arrow-down-tray" class="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Run Diagnostic Modal -->
    <URunDiagnosticModal
      v-model="showRunModal"
      :loading="isRunning"
      @submit="startDiagnostic"
      @cancel="showRunModal = false"
    />

    <!-- Results Modal -->
    <UModal
      v-if="selectedResult"
      v-model="showResultsModal"
      :title="t('diagnostics.results.titleWithName', { title: t('diagnostics.results.title'), name: selectedResult.name })"
      :description="t('diagnostics.results.subtitle')"
      size="xl"
    >
      <div class="space-y-6">
        <!-- Summary Cards -->
        <div class="grid gap-4 sm:grid-cols-3">
          <div class="u-card p-4 text-center">
            <div class="text-2xl font-bold text-green-600">
              {{ selectedResult.score }}/100
            </div>
            <p class="u-body-sm text-gray-500">{{ t('diagnostics.healthScore') }}</p>
          </div>
          <div class="u-card p-4 text-center">
            <div class="text-2xl font-bold text-gray-900">
              {{ selectedResult.issuesFound }}
            </div>
            <p class="u-body-sm text-gray-500">{{ t('diagnostics.issuesFound') }}</p>
          </div>
          <div class="u-card p-4 text-center">
            <div class="text-2xl font-bold text-gray-900">
              {{ selectedResult.duration }}
            </div>
            <p class="u-body-sm text-gray-500">{{ t('diagnostics.analysisDuration') }}</p>
          </div>
        </div>

        <!-- Recommendations -->
        <div class="u-card p-4 sm:p-6">
          <h4 class="u-h5 mb-4">Рекомендации</h4>
          <div class="space-y-4">
            <div class="p-4 border border-yellow-200 bg-yellow-50 rounded-lg">
              <div class="flex items-start gap-3">
                <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                <div class="min-w-0">
                  <p class="font-medium text-yellow-800">
                    {{ t('diagnostics.recommendations.pressureMaintenance') }}
                  </p>
                  <p class="u-body-sm text-yellow-700 mt-1">
                    {{ t('diagnostics.recommendations.pressureMaintenanceDesc') }}
                  </p>
                  <p class="text-xs text-yellow-600 mt-2">
                    {{ t('diagnostics.priority') }}: {{ t('diagnostics.priorityMedium') }}
                  </p>
                </div>
              </div>
            </div>

            <div class="p-4 border border-green-200 bg-green-50 rounded-lg">
              <div class="flex items-start gap-3">
                <Icon name="heroicons:check-circle" class="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                <div class="min-w-0">
                  <p class="font-medium text-green-800">
                    {{ t('diagnostics.recommendations.temperatureMonitoring') }}
                  </p>
                  <p class="u-body-sm text-green-700 mt-1">
                    {{ t('diagnostics.recommendations.temperatureMonitoringDesc') }}
                  </p>
                  <p class="text-xs text-green-600 mt-2">
                    {{ t('diagnostics.statusLabel') }}: {{ t('diagnostics.statusNormal') }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <template #footer>
        <button @click="showResultsModal = false" class="u-btn u-btn-secondary flex-1">
          {{ t('ui.close') }}
        </button>
        <button class="u-btn u-btn-primary flex-1">
          <Icon name="heroicons:arrow-down-tray" class="w-4 h-4 mr-2" />
          {{ t('diagnostics.exportPDF') }}
        </button>
      </template>
    </UModal>
  </div>
</template>

<script setup lang="ts">
import type { DiagnosticSession } from '~/types/api'

interface ActiveSession {
  id: number
  name: string
  equipment: string
  progress: number
  startedAt: string
}

interface DiagnosticUIResult {
  id: number
  name: string
  equipment: string
  score: number
  issuesFound: number
  completedAt: string
  status: 'completed' | 'warning' | 'error' | 'processing'
  duration: string
}

definePageMeta({ middleware: ['auth'] })
const { t } = useI18n()

const showRunModal = ref(false)
const showResultsModal = ref(false)
const isRunning = ref(false)
const selectedResult = ref<DiagnosticUIResult | null>(null)

const activeSessions = ref<ActiveSession[]>([])
const recentResults = ref<DiagnosticUIResult[]>([
  {
    id: 1,
    name: 'Full System Analysis - HYD-001',
    equipment: 'HYD-001 - Pump Station A',
    score: 92,
    issuesFound: 1,
    completedAt: '2 hours ago',
    status: 'completed',
    duration: '4.2 min'
  },
  {
    id: 2,
    name: 'Pressure Check - HYD-002',
    equipment: 'HYD-002 - Hydraulic Motor B',
    score: 78,
    issuesFound: 3,
    completedAt: '6 hours ago',
    status: 'warning',
    duration: '2.8 min'
  },
  {
    id: 3,
    name: 'Temperature Analysis - HYD-003',
    equipment: 'HYD-003 - Control Valve C',
    score: 95,
    issuesFound: 0,
    completedAt: '1 day ago',
    status: 'completed',
    duration: '3.1 min'
  }
])

const startDiagnostic = async (data: any) => {
  isRunning.value = true
  
  const session: ActiveSession = {
    id: Date.now(),
    name: `New Diagnostic - ${data.equipment}`,
    equipment: data.equipment,
    progress: 0,
    startedAt: 'now'
  }
  
  activeSessions.value.push(session)
  showRunModal.value = false
  
  const interval = setInterval(() => {
    session.progress += Math.random() * 20
    if (session.progress >= 100) {
      session.progress = 100
      clearInterval(interval)
      
      setTimeout(() => {
        activeSessions.value = activeSessions.value.filter(s => s.id !== session.id)
        isRunning.value = false
        
        const newResult: DiagnosticUIResult = {
          id: session.id,
          name: session.name,
          equipment: session.equipment,
          score: Math.floor(Math.random() * 30) + 70,
          issuesFound: Math.floor(Math.random() * 4),
          completedAt: 'just now',
          status: 'completed',
          duration: `${Math.floor(Math.random() * 3) + 2}.${Math.floor(Math.random() * 9)} min`
        }
        recentResults.value.unshift(newResult)
      }, 1000)
    }
  }, 500)
}

const cancelSession = (sessionId: number) => {
  activeSessions.value = activeSessions.value.filter(s => s.id !== sessionId)
  if (activeSessions.value.length === 0) {
    isRunning.value = false
  }
}

const viewResult = (resultId: number) => {
  const result = recentResults.value.find(r => r.id === resultId)
  if (result) {
    selectedResult.value = result
    showResultsModal.value = true
  }
}

const getStatusBadgeClass = (status: string): string => {
  const classes: Record<string, string> = {
    completed: 'u-badge-success',
    warning: 'u-badge-warning',
    error: 'u-badge-error',
    processing: 'u-badge-info'
  }
  return classes[status] || 'u-badge-gray'
}

const getStatusIcon = (status: string): string => {
  const icons: Record<string, string> = {
    completed: 'heroicons:check-circle',
    warning: 'heroicons:exclamation-triangle',
    error: 'heroicons:x-circle',
    processing: 'heroicons:arrow-path'
  }
  return icons[status] || 'heroicons:question-mark-circle'
}
</script>