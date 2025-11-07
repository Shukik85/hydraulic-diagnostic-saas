<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="u-h2">{{ t('dashboard.title') }}</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">{{ t('dashboard.subtitle') }}</p>
      </div>
      <div class="flex items-center gap-3">
        <div class="u-flex-center gap-2 px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800">
          <Icon name="heroicons:clock" class="w-4 h-4 text-gray-600 dark:text-gray-400" />
          <span class="text-sm text-gray-600 dark:text-gray-400">{{ t('dashboard.liveStatus') }}</span>
          <span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
        </div>
        <button class="u-btn u-btn-secondary u-btn-md">
          <Icon name="heroicons:arrow-path" class="w-4 h-4 mr-2" />
          {{ t('dashboard.refreshBtn') }}
        </button>
      </div>
    </div>

    <!-- KPI Metrics Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <!-- Active Systems -->
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('dashboard.kpi.activeSystems') }}</h3>
          <div class="u-metric-icon bg-blue-100 dark:bg-blue-900/30">
            <Icon name="heroicons:server-stack" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <div class="u-metric-value">127</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>+5 {{ t('dashboard.kpi.fromYesterday') }}</span>
        </div>
      </div>

      <!-- System Health -->
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('dashboard.kpi.systemHealth') }}</h3>
          <div class="u-metric-icon bg-green-100 dark:bg-green-900/30">
            <Icon name="heroicons:heart" class="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <div class="u-metric-value">99.9%</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>{{ t('dashboard.kpi.uptimeExcellence') }}</span>
        </div>
      </div>

      <!-- Prevented Failures -->
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('dashboard.kpi.preventedFailures') }}</h3>
          <div class="u-metric-icon bg-purple-100 dark:bg-purple-900/30">
            <Icon name="heroicons:shield-check" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
          </div>
        </div>
        <div class="u-metric-value">23</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>{{ t('dashboard.kpi.aiPredictions') }}</span>
        </div>
      </div>

      <!-- Cost Savings -->
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">{{ t('dashboard.kpi.costSavings') }}</h3>
          <div class="u-metric-icon bg-orange-100 dark:bg-orange-900/30">
            <Icon name="heroicons:currency-dollar" class="w-5 h-5 text-orange-600 dark:text-orange-400" />
          </div>
        </div>
        <div class="u-metric-value">89%</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>{{ t('dashboard.kpi.vsPreviousYear') }}</span>
        </div>
      </div>
    </div>

    <!-- Charts Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- System Performance Chart -->
      <div class="u-chart-wrapper">
        <div class="u-chart-header">
          <h3 class="u-chart-title">{{ t('dashboard.charts.systemPerformance') }}</h3>
          <div class="u-chart-controls">
            <select class="u-input text-sm py-1 px-2 w-32">
              <option>{{ t('dashboard.charts.last24h') }}</option>
              <option>{{ t('dashboard.charts.last7d') }}</option>
              <option>{{ t('dashboard.charts.last30d') }}</option>
            </select>
          </div>
        </div>
        <div class="u-chart-container">
          <ClientOnly>
            <component 
              :is="VChart"
              :option="performanceChartOption"
              :autoresize="true"
              class="w-full h-full"
            />
            <template #fallback>
              <div class="u-flex-center h-full">
                <div class="u-spinner w-8 h-8"></div>
              </div>
            </template>
          </ClientOnly>
        </div>
      </div>

      <!-- AI Predictions Chart -->
      <div class="u-chart-wrapper">
        <div class="u-chart-header">
          <h3 class="u-chart-title">{{ t('dashboard.charts.aiPredictions') }}</h3>
          <span class="u-badge u-badge-info">
            <Icon name="heroicons:sparkles" class="w-4 h-4" />
            {{ t('dashboard.charts.mlActive') }}
          </span>
        </div>
        <div class="u-chart-container">
          <ClientOnly>
            <component 
              :is="VChart"
              :option="aiChartOption"
              :autoresize="true"
              class="w-full h-full"
            />
            <template #fallback>
              <div class="u-flex-center h-full">
                <div class="u-spinner w-8 h-8"></div>
              </div>
            </template>
          </ClientOnly>
        </div>
      </div>
    </div>

    <!-- Recent Activity & Quick Actions -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <!-- Quick Actions -->
      <div class="u-card p-6">
        <h3 class="u-h4 mb-6">{{ t('dashboard.quickActions.title') }}</h3>
        <div class="space-y-3">
          <button @click="openDiagnosticModal = true" class="w-full flex items-center gap-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 u-transition-fast text-left">
            <Icon name="heroicons:play" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span class="text-sm font-medium text-gray-900 dark:text-white">{{ t('dashboard.quickActions.runDiagnostics') }}</span>
          </button>
          <button @click="openReportModal = true" class="w-full flex items-center gap-3 p-3 rounded-lg bg-green-50 dark:bg-green-900/30 hover:bg-green-100 dark:hover:bg-green-900/50 u-transition-fast text-left">
            <Icon name="heroicons:document-text" class="w-5 h-5 text-green-600 dark:text-green-400" />
            <span class="text-sm font-medium text-gray-900 dark:text-white">{{ t('dashboard.quickActions.generateReport') }}</span>
          </button>
          <button @click="openSystemModal = true" class="w-full flex items-center gap-3 p-3 rounded-lg bg-purple-50 dark:bg-purple-900/30 hover:bg-purple-100 dark:hover:bg-purple-900/50 u-transition-fast text-left">
            <Icon name="heroicons:plus" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <span class="text-sm font-medium text-gray-900 dark:text-white">{{ t('dashboard.quickActions.addSystem') }}</span>
          </button>
        </div>
      </div>

      <!-- Recent Events -->
      <div class="u-card p-6">
        <h3 class="u-h4 mb-6">{{ t('dashboard.recentEvents.title') }}</h3>
        <div class="space-y-3">
          <div class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
            <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-white">
                {{ t('dashboard.recentEvents.systemNormal', ['127']) }}
              </p>
              <p class="text-xs text-gray-600 dark:text-gray-400">{{ t('dashboard.recentEvents.minutesAgo', ['2']) }}</p>
            </div>
          </div>
          <div class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
            <div class="w-2 h-2 bg-yellow-500 rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-white">
                {{ t('dashboard.recentEvents.anomalyDetected', ['89']) }}
              </p>
              <p class="text-xs text-gray-600 dark:text-gray-400">{{ t('dashboard.recentEvents.minutesAgo', ['15']) }}</p>
            </div>
          </div>
          <div class="flex items-center gap-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
            <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-white">
                {{ t('dashboard.recentEvents.diagnosticsCompleted', ['45']) }}
              </p>
              <p class="text-xs text-gray-600 dark:text-gray-400">{{ t('dashboard.recentEvents.hourAgo', ['1']) }}</p>
            </div>
          </div>
        </div>
      </div>

      <!-- AI Status -->
      <div class="u-card p-6">
        <h3 class="u-h4 mb-6">{{ t('dashboard.aiStatus.title') }}</h3>
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="u-body text-gray-600 dark:text-gray-400">{{ t('dashboard.aiStatus.diagnosticModel') }}</span>
            <span class="u-badge u-badge-success">
              <Icon name="heroicons:check-circle" class="w-3 h-3" />
              {{ t('dashboard.aiStatus.active') }}
            </span>
          </div>
          <div class="flex items-center justify-between">
            <span class="u-body text-gray-600 dark:text-gray-400">{{ t('dashboard.aiStatus.predictionAccuracy') }}</span>
            <span class="text-sm font-semibold text-blue-600 dark:text-blue-400">97.3%</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="u-body text-gray-600 dark:text-gray-400">{{ t('dashboard.aiStatus.modelTraining') }}</span>
            <span class="u-badge u-badge-info">
              <Icon name="heroicons:academic-cap" class="w-3 h-3" />
              {{ t('dashboard.aiStatus.complete') }}
            </span>
          </div>
          <div class="u-divider"></div>
          <div class="flex items-center justify-between">
            <span class="u-body text-gray-600 dark:text-gray-400">{{ t('dashboard.aiStatus.lastUpdate') }}</span>
            <span class="text-xs text-gray-500 dark:text-gray-500">{{ t('dashboard.aiStatus.minAgo', ['5']) }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Modal Components -->
    <URunDiagnosticModal 
      v-model="openDiagnosticModal" 
      :loading="diagnosticLoading"
      @submit="onRunDiagnostic"
      @cancel="onCancelDiagnostic"
    />

    <UReportGenerateModal 
      v-model="openReportModal" 
      :loading="reportLoading"
      @submit="onGenerateReport"
      @cancel="onCancelReport"
    />

    <UCreateSystemModal 
      v-model="openSystemModal" 
      :loading="systemLoading"
      @submit="onCreateSystem"
      @cancel="onCancelSystem"
    />
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'dashboard',
  title: 'Enterprise Dashboard',
  middleware: ['auth']
})

const { t } = useI18n()
const { $VChart: VChart } = useNuxtApp()

// Modal states
const openDiagnosticModal = ref(false)
const openReportModal = ref(false)
const openSystemModal = ref(false)
const diagnosticLoading = ref(false)
const reportLoading = ref(false)
const systemLoading = ref(false)

// Modal handlers
const onRunDiagnostic = async (data: any) => {
  diagnosticLoading.value = true
  try {
    console.log('Starting diagnostic with data:', data)
    await new Promise(resolve => setTimeout(resolve, 2000))
    openDiagnosticModal.value = false
    alert('Diagnostic started successfully! Check /diagnostics for progress.')
  } catch (error: any) {
    alert(`Failed to start diagnostic: ${error?.message || 'Unknown error'}`)
  } finally {
    diagnosticLoading.value = false
  }
}

const onCancelDiagnostic = () => { openDiagnosticModal.value = false }

const onGenerateReport = async (data: any) => {
  reportLoading.value = true
  try {
    console.log('Generating report with data:', data)
    await new Promise(resolve => setTimeout(resolve, 2000))
    openReportModal.value = false
    alert('Report generation started! Check /reports when complete.')
  } catch (error: any) {
    alert(`Failed to generate report: ${error?.message || 'Unknown error'}`)
  } finally {
    reportLoading.value = false
  }
}

const onCancelReport = () => { openReportModal.value = false }

const onCreateSystem = async (data: any) => {
  systemLoading.value = true
  try {
    console.log('Creating system with data:', data)
    await new Promise(resolve => setTimeout(resolve, 1500))
    openSystemModal.value = false
    alert('System created successfully! Check /systems to configure.')
  } catch (error: any) {
    alert(`Failed to create system: ${error?.message || 'Unknown error'}`)
  } finally {
    systemLoading.value = false
  }
}

const onCancelSystem = () => { openSystemModal.value = false }

// Performance Chart with Enterprise styling
const performanceChartOption = ref({
  backgroundColor: 'transparent',
  textStyle: { fontFamily: 'var(--font-sans)', fontSize: 12 },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e5e7eb',
    borderRadius: 8,
    textStyle: { color: '#374151', fontSize: 13 }
  },
  legend: {
    data: ['Performance', 'Efficiency'],
    textStyle: { color: '#6b7280', fontSize: 12 },
    top: 10
  },
  grid: { left: '3%', right: '4%', bottom: '15%', top: '15%', containLabel: true },
  xAxis: {
    type: 'category',
    data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'],
    axisLine: { lineStyle: { color: '#e5e7eb' } },
    axisLabel: { color: '#6b7280', fontSize: 11 },
    axisTick: { show: false }
  },
  yAxis: {
    type: 'value',
    name: '%',
    axisLine: { show: false },
    splitLine: { lineStyle: { color: '#f3f4f6', type: 'dashed' } },
    axisLabel: { color: '#6b7280', fontSize: 11 }
  },
  series: [
    {
      name: 'Performance',
      type: 'line',
      data: [94, 96, 98, 95, 97, 99, 94],
      smooth: true,
      lineStyle: { color: '#2563eb', width: 3 },
      itemStyle: { color: '#2563eb', borderWidth: 2 },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0, y: 0, x2: 0, y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(37, 99, 235, 0.15)' },
            { offset: 1, color: 'rgba(37, 99, 235, 0.02)' }
          ]
        }
      }
    },
    {
      name: 'Efficiency',
      type: 'line',
      data: [87, 89, 91, 88, 92, 94, 89],
      smooth: true,
      lineStyle: { color: '#10b981', width: 2, type: 'dashed' },
      itemStyle: { color: '#10b981' }
    }
  ]
})

// AI Predictions Chart with Enterprise styling
const aiChartOption = ref({
  backgroundColor: 'transparent',
  textStyle: { fontFamily: 'var(--font-sans)', fontSize: 12 },
  tooltip: {
    trigger: 'item',
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderColor: '#e5e7eb',
    borderRadius: 8,
    textStyle: { color: '#374151', fontSize: 13 }
  },
  legend: {
    orient: 'vertical',
    left: 'left',
    textStyle: { color: '#6b7280', fontSize: 12 }
  },
  series: [{
    name: 'Predictions',
    type: 'pie',
    radius: ['40%', '70%'],
    center: ['60%', '50%'],
    avoidLabelOverlap: false,
    itemStyle: {
      borderRadius: 8,
      borderColor: '#fff',
      borderWidth: 2
    },
    label: {
      show: false,
      position: 'center'
    },
    emphasis: {
      label: {
        show: true,
        fontSize: 14,
        fontWeight: 'bold'
      }
    },
    labelLine: { show: false },
    data: [
      { value: 89, name: t('dashboard.charts.normalOps'), itemStyle: { color: '#10b981' } },
      { value: 8, name: t('dashboard.charts.maintenanceSoon'), itemStyle: { color: '#f59e0b' } },
      { value: 3, name: t('dashboard.charts.criticalIssues'), itemStyle: { color: '#ef4444' } }
    ]
  }]
})
</script>
