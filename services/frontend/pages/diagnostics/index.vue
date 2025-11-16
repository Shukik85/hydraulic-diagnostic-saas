<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ t('diagnostics.title') }}</h1>
        <p class="text-steel-shine mt-2">
          {{ t('diagnostics.subtitle') }}
        </p>
      </div>
      <UButton 
        size="lg"
        @click="showRunModal = true"
      >
        <Icon name="heroicons:play" class="w-5 h-5 mr-2" />
        {{ t('diagnostics.runNew') }}
      </UButton>
    </div>

    <!-- Zero State - показываем только если нет данных -->
    <UZeroState
      v-if="!loading && recentResults.length === 0 && activeSessions.length === 0"
      icon-name="heroicons:document-magnifying-glass"
      :title="t('diagnostics.empty.title')"
      :description="t('diagnostics.empty.description')"
      action-icon="heroicons:play"
      :action-text="t('diagnostics.empty.action')"
      @action="showRunModal = true"
    />

    <!-- Content - показываем только если есть данные -->
    <template v-else>
      <!-- KPI Overview -->
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <!-- Active Sessions KPI -->
        <div class="card-glass p-6">
          <div class="flex items-start justify-between mb-4">
            <div class="flex-1">
              <p class="text-sm text-steel-shine font-medium mb-1">
                {{ t('diagnostics.kpi.activeSessions') }}
              </p>
              <div class="text-4xl font-bold text-white">
                {{ activeSessions.length }}
              </div>
            </div>
            <div class="w-12 h-12 rounded-lg bg-primary-600/10 flex items-center justify-center">
              <Icon name="heroicons:play-circle" class="w-6 h-6 text-primary-400" />
            </div>
          </div>
          <div class="flex items-center gap-1.5 text-steel-400">
            <Icon name="heroicons:clock" class="w-4 h-4" />
            <span class="text-sm">{{ t('diagnostics.kpi.runningNow') }}</span>
          </div>
        </div>

        <!-- Success Rate KPI -->
        <div class="card-glass p-6">
          <div class="flex items-start justify-between mb-4">
            <div class="flex-1">
              <p class="text-sm text-steel-shine font-medium mb-1">
                {{ t('diagnostics.kpi.successRate') }}
              </p>
              <div class="text-4xl font-bold text-white">
                98.5<span class="text-lg text-steel-400">%</span>
              </div>
            </div>
            <div class="w-12 h-12 rounded-lg bg-success-600/10 flex items-center justify-center">
              <Icon name="heroicons:check-badge" class="w-6 h-6 text-success-400" />
            </div>
          </div>
          <div class="flex items-center gap-1.5 text-success-400">
            <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
            <span class="text-sm font-medium">+2.1%</span>
            <span class="text-xs text-steel-400">{{ t('diagnostics.kpi.thisWeek') }}</span>
          </div>
        </div>

        <!-- Avg Duration KPI -->
        <div class="card-glass p-6">
          <div class="flex items-start justify-between mb-4">
            <div class="flex-1">
              <p class="text-sm text-steel-shine font-medium mb-1">
                {{ t('diagnostics.kpi.avgDuration') }}
              </p>
              <div class="flex items-baseline gap-1">
                <span class="text-4xl font-bold text-white">4.2</span>
                <span class="text-lg text-steel-400">{{ t('ui.minutes') }}</span>
              </div>
            </div>
            <div class="w-12 h-12 rounded-lg bg-purple-600/10 flex items-center justify-center">
              <Icon name="heroicons:clock" class="w-6 h-6 text-purple-400" />
            </div>
          </div>
          <div class="flex items-center gap-1.5 text-success-400">
            <Icon name="heroicons:arrow-trending-down" class="w-4 h-4" />
            <span class="text-sm font-medium">-0.8</span>
            <span class="text-xs text-steel-400">{{ t('diagnostics.kpi.faster') }}</span>
          </div>
        </div>

        <!-- Issues Found KPI -->
        <div class="card-glass p-6">
          <div class="flex items-start justify-between mb-4">
            <div class="flex-1">
              <p class="text-sm text-steel-shine font-medium mb-1">
                {{ t('diagnostics.kpi.issuesFound') }}
              </p>
              <div class="text-4xl font-bold text-white">
                7
              </div>
            </div>
            <div class="w-12 h-12 rounded-lg bg-red-600/10 flex items-center justify-center">
              <Icon name="heroicons:exclamation-triangle" class="w-6 h-6 text-red-400" />
            </div>
          </div>
          <div class="flex items-center gap-1.5 text-yellow-400">
            <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
            <span class="text-sm">{{ t('diagnostics.kpi.needsAttention') }}</span>
          </div>
        </div>
      </div>

      <!-- Active Sessions -->
      <div v-if="activeSessions.length > 0">
        <h2 class="text-xl font-bold text-white mb-6">{{ t('diagnostics.activeSessions') }}</h2>
        <div class="space-y-4">
          <div 
            v-for="session in activeSessions" 
            :key="session.id" 
            class="card-glass p-6"
          >
            <div class="flex flex-col sm:flex-row sm:items-center gap-4">
              <!-- Session Info -->
              <div class="flex items-center gap-4 flex-1 min-w-0">
                <UStatusDot status="info" :animated="true" />
                <div class="min-w-0">
                  <p class="font-medium text-white truncate">{{ session.name }}</p>
                  <p class="text-sm text-steel-shine truncate">
                    {{ session.equipment }} • {{ t('diagnostics.started') }} {{ session.startedAt }}
                  </p>
                </div>
              </div>

              <!-- Progress -->
              <div class="flex items-center gap-4 justify-between sm:justify-end">
                <div class="w-24 sm:w-32">
                  <div class="flex items-center gap-2">
                    <div class="flex-1 progress-bar">
                      <div 
                        class="progress-fill" 
                        :style="{ width: session.progress + '%' }"
                      />
                    </div>
                    <span class="text-sm font-medium text-white">
                      {{ Math.round(session.progress) }}%
                    </span>
                  </div>
                </div>
                <UButton 
                  variant="ghost" 
                  size="sm"
                  @click="cancelSession(session.id)"
                >
                  <Icon name="heroicons:x-mark" class="w-4 h-4 mr-1" />
                  {{ t('ui.cancel') }}
                </UButton>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Recent Results -->
      <UCard class="overflow-hidden">
        <UCardHeader class="border-b border-steel-700/50">
          <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <UCardTitle>{{ t('diagnostics.recentResults.title') }}</UCardTitle>
              <p class="text-steel-shine mt-1">
                {{ t('diagnostics.recentResults.subtitle') }}
              </p>
            </div>
            <div class="flex items-center gap-2">
              <USelect class="w-full sm:w-40">
                <option>{{ t('diagnostics.filters.allEquipment') }}</option>
                <option>HYD-001</option>
                <option>HYD-002</option>
                <option>HYD-003</option>
              </USelect>
              <UButton variant="ghost" size="icon">
                <Icon name="heroicons:funnel" class="w-5 h-5" />
              </UButton>
            </div>
          </div>
        </UCardHeader>

        <UCardContent class="p-0">
          <!-- Mobile: Card Layout -->
          <div class="sm:hidden space-y-4 p-4">
            <div 
              v-for="result in recentResults" 
              :key="result.id" 
              class="card-glass p-4"
            >
              <div class="flex items-start justify-between gap-3 mb-3">
                <div class="flex-1 min-w-0">
                  <h4 class="font-medium text-white truncate">{{ result.name }}</h4>
                  <p class="text-sm text-steel-shine">{{ result.equipment }}</p>
                </div>
                <UBadge :variant="getStatusVariant(result.status)">
                  <Icon :name="getStatusIcon(result.status)" class="w-3 h-3" />
                  {{ t(`diagnostics.status.${result.status}`) }}
                </UBadge>
              </div>

              <div class="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <p class="text-xs text-steel-shine mb-1">{{ t('diagnostics.healthScore') }}</p>
                  <div class="flex items-center gap-2">
                    <div class="w-8 progress-bar">
                      <div 
                        :class="result.score >= 90 ? 'progress-fill-success' : result.score >= 70 ? 'progress-fill-warning' : 'progress-fill-error'"
                        :style="{ width: result.score + '%' }"
                      />
                    </div>
                    <span class="text-sm font-medium text-white">{{ result.score }}/100</span>
                  </div>
                </div>
                <div>
                  <p class="text-xs text-steel-shine mb-1">{{ t('diagnostics.issues') }}</p>
                  <UBadge
                    :variant="result.issuesFound === 0 ? 'success' : result.issuesFound <= 2 ? 'warning' : 'destructive'"
                  >
                    {{ result.issuesFound }} {{ t('diagnostics.issuesCount') }}
                  </UBadge>
                </div>
              </div>

              <div class="flex items-center justify-between">
                <span class="text-sm text-steel-shine">{{ result.completedAt }}</span>
                <div class="flex items-center gap-2">
                  <UButton 
                    variant="ghost" 
                    size="icon"
                    @click="viewResult(result.id)"
                  >
                    <Icon name="heroicons:eye" class="w-5 h-5" />
                  </UButton>
                  <UButton variant="ghost" size="icon">
                    <Icon name="heroicons:arrow-down-tray" class="w-5 h-5" />
                  </UButton>
                </div>
              </div>
            </div>
          </div>

          <!-- Desktop: Table Layout -->
          <div class="hidden sm:block overflow-x-auto">
            <table class="w-full">
              <thead class="bg-steel-900/50 border-b border-steel-700/50">
                <tr>
                  <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">
                    {{ t('diagnostics.table.name') }}
                  </th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">
                    {{ t('diagnostics.table.equipment') }}
                  </th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">
                    {{ t('diagnostics.table.healthScore') }}
                  </th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">
                    {{ t('diagnostics.table.issues') }}
                  </th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">
                    {{ t('diagnostics.table.completed') }}
                  </th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">
                    {{ t('diagnostics.table.status') }}
                  </th>
                  <th class="px-6 py-3 text-left text-xs font-medium text-steel-shine uppercase tracking-wider">
                    {{ t('diagnostics.table.actions') }}
                  </th>
                </tr>
              </thead>
              <tbody class="divide-y divide-steel-700/50">
                <tr 
                  v-for="result in recentResults" 
                  :key="result.id"
                  class="hover:bg-steel-900/30 transition-colors"
                >
                  <td class="px-6 py-4 whitespace-nowrap font-medium text-white">{{ result.name }}</td>
                  <td class="px-6 py-4 whitespace-nowrap text-steel-shine">{{ result.equipment }}</td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center gap-2">
                      <div class="w-12 progress-bar">
                        <div 
                          :class="result.score >= 90 ? 'progress-fill-success' : result.score >= 70 ? 'progress-fill-warning' : 'progress-fill-error'"
                          :style="{ width: result.score + '%' }"
                        />
                      </div>
                      <span class="text-sm font-medium text-white">{{ result.score }}/100</span>
                    </div>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <UBadge
                      :variant="result.issuesFound === 0 ? 'success' : result.issuesFound <= 2 ? 'warning' : 'destructive'"
                    >
                      {{ result.issuesFound }} {{ t('diagnostics.issuesCount') }}
                    </UBadge>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap text-sm text-steel-shine">{{ result.completedAt }}</td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <UBadge :variant="getStatusVariant(result.status)">
                      <Icon :name="getStatusIcon(result.status)" class="w-3 h-3" />
                      {{ t(`diagnostics.status.${result.status}`) }}
                    </UBadge>
                  </td>
                  <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center gap-2">
                      <UButton 
                        variant="ghost" 
                        size="icon"
                        @click="viewResult(result.id)"
                      >
                        <Icon name="heroicons:eye" class="w-5 h-5" />
                      </UButton>
                      <UButton variant="ghost" size="icon">
                        <Icon name="heroicons:arrow-down-tray" class="w-5 h-5" />
                      </UButton>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </UCardContent>
      </UCard>
    </template>

    <!-- Run Diagnostic Modal -->
    <URunDiagnosticModal 
      v-model="showRunModal" 
      :loading="isRunning" 
      @submit="startDiagnostic"
    />

    <!-- Results Modal -->
    <UDialog v-model="showResultsModal">
      <UDialogContent class="max-w-3xl">
        <UDialogHeader>
          <UDialogTitle>
            {{ selectedResult?.name || t('diagnostics.results.title') }}
          </UDialogTitle>
          <UDialogDescription>
            {{ t('diagnostics.results.subtitle') }}
          </UDialogDescription>
        </UDialogHeader>

        <div v-if="selectedResult" class="space-y-6">
          <!-- Summary Cards -->
          <div class="grid gap-4 sm:grid-cols-3">
            <div class="card-glass p-6 text-center">
              <div class="text-3xl font-bold text-success-400">
                {{ selectedResult.score }}/100
              </div>
              <p class="text-sm text-steel-shine mt-2">{{ t('diagnostics.healthScore') }}</p>
            </div>
            <div class="card-glass p-6 text-center">
              <div class="text-3xl font-bold text-white">
                {{ selectedResult.issuesFound }}
              </div>
              <p class="text-sm text-steel-shine mt-2">{{ t('diagnostics.issuesFound') }}</p>
            </div>
            <div class="card-glass p-6 text-center">
              <div class="text-3xl font-bold text-white">
                {{ selectedResult.duration }}
              </div>
              <p class="text-sm text-steel-shine mt-2">{{ t('diagnostics.analysisDuration') }}</p>
            </div>
          </div>

          <!-- Recommendations -->
          <div class="card-glass p-6">
            <h4 class="text-lg font-bold text-white mb-4">{{ t('diagnostics.recommendations.title') }}</h4>
            <div class="space-y-4">
              <div class="alert-warning">
                <Icon name="heroicons:exclamation-triangle" class="w-5 h-5" />
                <div>
                  <p class="font-medium">
                    {{ t('diagnostics.recommendations.pressureMaintenance') }}
                  </p>
                  <p class="text-sm mt-1">
                    {{ t('diagnostics.recommendations.pressureMaintenanceDesc') }}
                  </p>
                  <p class="text-xs mt-2 opacity-75">
                    {{ t('diagnostics.priority') }}: {{ t('diagnostics.priorityMedium') }}
                  </p>
                </div>
              </div>

              <div class="alert-success">
                <Icon name="heroicons:check-circle" class="w-5 h-5" />
                <div>
                  <p class="font-medium">
                    {{ t('diagnostics.recommendations.temperatureMonitoring') }}
                  </p>
                  <p class="text-sm mt-1">
                    {{ t('diagnostics.recommendations.temperatureMonitoringDesc') }}
                  </p>
                  <p class="text-xs mt-2 opacity-75">
                    {{ t('diagnostics.statusLabel') }}: {{ t('diagnostics.statusNormal') }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <UDialogFooter>
          <UButton 
            variant="secondary"
            @click="showResultsModal = false"
          >
            {{ t('ui.close') }}
          </UButton>
          <UButton>
            <Icon name="heroicons:arrow-down-tray" class="w-5 h-5 mr-2" />
            {{ t('diagnostics.exportPDF') }}
          </UButton>
        </UDialogFooter>
      </UDialogContent>
    </UDialog>
  </div>
</template>

<script setup lang="ts">
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
const loading = ref(false)
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
    showResultsModal = true
  }
}

const getStatusVariant = (status: string): 'default' | 'success' | 'warning' | 'destructive' => {
  const variants: Record<string, 'default' | 'success' | 'warning' | 'destructive'> = {
    completed: 'success',
    warning: 'warning',
    error: 'destructive',
    processing: 'default'
  }
  return variants[status] || 'default'
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
