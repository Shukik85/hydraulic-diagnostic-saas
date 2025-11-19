<template>
  <div class="space-y-10">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ t('dashboard.title') }}</h1>
        <p class="text-steel-shine mt-2">{{ t('dashboard.subtitle') }}</p>
      </div>
      <div class="flex items-center gap-3">
        <!-- Live Status Badge -->
        <div class="flex gap-2 items-center px-4 py-2 rounded-lg card-glass border border-steel-700/50">
          <Icon name="heroicons:signal" class="w-4 h-4 text-success-400" />
          <span class="text-sm text-white">{{ t('dashboard.liveStatus') }}</span>
          <UStatusDot status="success" :animated="true" />
        </div>
        <UButton 
          variant="secondary"
          @click="refreshData"
        >
          <Icon name="heroicons:arrow-path" class="w-5 h-5 mr-2" />
          {{ t('dashboard.refreshBtn') }}
        </UButton>
      </div>
    </div>

    <!-- KPI Grid using KpiCard -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <KpiCard
        :title="t('dashboard.kpi.activeSystems')"
        :value="127"
        icon="heroicons:server-stack"
        color="primary"
        :growth="3.9"
        :description="t('dashboard.kpi.fromYesterday')"
      />

      <KpiCard
        :title="t('dashboard.kpi.systemHealth')"
        value="99.9%"
        icon="heroicons:heart"
        color="success"
        :growth="0.1"
        :description="t('dashboard.kpi.uptimeExcellence')"
      />

      <KpiCard
        :title="t('dashboard.kpi.preventedFailures')"
        :value="23"
        icon="heroicons:shield-check"
        color="info"
        :growth="15.2"
        :description="t('dashboard.kpi.aiPredictions')"
      />

      <KpiCard
        :title="t('dashboard.kpi.costSavings')"
        value="89%"
        icon="heroicons:currency-dollar"
        color="warning"
        :growth="12.5"
        :description="t('dashboard.kpi.vsPreviousYear')"
      />
    </div>

    <!-- Charts Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- System Performance Chart -->
      <UCard>
        <UCardHeader class="border-b border-steel-700/50">
          <div class="flex justify-between items-center">
            <UCardTitle>{{ t('dashboard.charts.systemPerformance') }}</UCardTitle>
            <USelect v-model="performancePeriod" class="w-40">
              <option value="24h">{{ t('dashboard.charts.last24h') }}</option>
              <option value="7d">{{ t('dashboard.charts.last7d') }}</option>
              <option value="30d">{{ t('dashboard.charts.last30d') }}</option>
            </USelect>
          </div>
        </UCardHeader>
        <UCardContent class="p-6">
          <div class="h-[300px]">
            <ClientOnly>
              <component 
                :is="VChart"
                :option="performanceChartOption"
                :autoresize="true"
                class="w-full h-full"
              />
              <template #fallback>
                <div class="flex items-center justify-center h-full">
                  <div class="w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
                </div>
              </template>
            </ClientOnly>
          </div>
        </UCardContent>
      </UCard>

      <!-- AI Predictions Chart -->
      <UCard>
        <UCardHeader class="border-b border-steel-700/50">
          <div class="flex justify-between items-center">
            <UCardTitle>{{ t('dashboard.charts.aiPredictions') }}</UCardTitle>
            <UBadge variant="default">
              <Icon name="heroicons:sparkles" class="w-4 h-4" />
              {{ t('dashboard.charts.mlActive') }}
            </UBadge>
          </div>
        </UCardHeader>
        <UCardContent class="p-6">
          <div class="h-[300px]">
            <ClientOnly>
              <component 
                :is="VChart"
                :option="aiChartOption"
                :autoresize="true"
                class="w-full h-full"
              />
              <template #fallback>
                <div class="flex items-center justify-center h-full">
                  <div class="w-8 h-8 border-4 border-purple-600 border-t-transparent rounded-full animate-spin" />
                </div>
              </template>
            </ClientOnly>
          </div>
        </UCardContent>
      </UCard>
    </div>

    <!-- Bottom Grid: Quick Actions, Recent Events, AI Status -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <!-- Quick Actions -->
      <UCard>
        <UCardHeader>
          <UCardTitle>{{ t('dashboard.quickActions.title') }}</UCardTitle>
        </UCardHeader>
        <UCardContent class="space-y-3 p-6">
          <button 
            class="w-full flex items-center gap-3 p-4 rounded-lg bg-primary-600/10 hover:bg-primary-600/20 border border-primary-500/30 hover:border-primary-500/50 transition-all text-left group"
            @click="openDiagnosticModal = true"
          >
            <div class="w-10 h-10 rounded-lg bg-primary-600/20 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Icon name="heroicons:play" class="w-5 h-5 text-primary-400" />
            </div>
            <span class="text-sm font-medium text-white">{{ t('dashboard.quickActions.runDiagnostics') }}</span>
          </button>

          <button 
            class="w-full flex items-center gap-3 p-4 rounded-lg bg-success-600/10 hover:bg-success-600/20 border border-success-500/30 hover:border-success-500/50 transition-all text-left group"
            @click="openReportModal = true"
          >
            <div class="w-10 h-10 rounded-lg bg-success-600/20 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Icon name="heroicons:document-text" class="w-5 h-5 text-success-400" />
            </div>
            <span class="text-sm font-medium text-white">{{ t('dashboard.quickActions.generateReport') }}</span>
          </button>

          <button 
            class="w-full flex items-center gap-3 p-4 rounded-lg bg-purple-600/10 hover:bg-purple-600/20 border border-purple-500/30 hover:border-purple-500/50 transition-all text-left group"
            @click="openSystemModal = true"
          >
            <div class="w-10 h-10 rounded-lg bg-purple-600/20 flex items-center justify-center group-hover:scale-110 transition-transform">
              <Icon name="heroicons:plus" class="w-5 h-5 text-purple-400" />
            </div>
            <span class="text-sm font-medium text-white">{{ t('dashboard.quickActions.addSystem') }}</span>
          </button>
        </UCardContent>
      </UCard>

      <!-- Recent Events -->
      <UCard>
        <UCardHeader>
          <UCardTitle>{{ t('dashboard.recentEvents.title') }}</UCardTitle>
        </UCardHeader>
        <UCardContent class="space-y-3 p-6">
          <div class="flex items-center gap-3 p-3 rounded-lg bg-success-500/5 border border-success-500/20">
            <UStatusDot status="success" :animated="true" />
            <div class="flex-1">
              <p class="text-sm font-medium text-white">
                {{ t('dashboard.recentEvents.systemNormal', ['127']) }}
              </p>
              <p class="text-xs text-steel-400">{{ t('dashboard.recentEvents.minutesAgo', ['2']) }}</p>
            </div>
          </div>

          <div class="flex items-center gap-3 p-3 rounded-lg bg-yellow-500/5 border border-yellow-500/20">
            <UStatusDot status="warning" />
            <div class="flex-1">
              <p class="text-sm font-medium text-white">
                {{ t('dashboard.recentEvents.anomalyDetected', ['89']) }}
              </p>
              <p class="text-xs text-steel-400">{{ t('dashboard.recentEvents.minutesAgo', ['15']) }}</p>
            </div>
          </div>

          <div class="flex items-center gap-3 p-3 rounded-lg bg-blue-500/5 border border-blue-500/20">
            <UStatusDot status="info" />
            <div class="flex-1">
              <p class="text-sm font-medium text-white">
                {{ t('dashboard.recentEvents.diagnosticsCompleted', ['45']) }}
              </p>
              <p class="text-xs text-steel-400">{{ t('dashboard.recentEvents.hourAgo', ['1']) }}</p>
            </div>
          </div>
        </UCardContent>
      </UCard>

      <!-- AI Status -->
      <UCard>
        <UCardHeader>
          <UCardTitle>{{ t('dashboard.aiStatus.title') }}</UCardTitle>
        </UCardHeader>
        <UCardContent class="space-y-4 p-6">
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.diagnosticModel') }}</span>
            <UBadge variant="success">
              <Icon name="heroicons:check-circle" class="w-3 h-3" />
              {{ t('dashboard.aiStatus.active') }}
            </UBadge>
          </div>

          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.predictionAccuracy') }}</span>
            <span class="text-sm font-semibold text-primary-400">97.3%</span>
          </div>

          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.modelTraining') }}</span>
            <UBadge variant="default">
              <Icon name="heroicons:check-badge" class="w-3 h-3" />
              {{ t('dashboard.aiStatus.complete') }}
            </UBadge>
          </div>

          <div class="border-t border-steel-700/50 my-4" />

          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.lastUpdate') }}</span>
            <span class="text-xs text-steel-400 flex items-center gap-1">
              <Icon name="heroicons:clock" class="w-3 h-3" />
              {{ t('dashboard.aiStatus.minAgo', ['5']) }}
            </span>
          </div>
        </UCardContent>
      </UCard>
    </div>

    <!-- Modals -->
    <URunDiagnosticModal 
      v-model="openDiagnosticModal" 
      :loading="diagnosticLoading" 
      @submit="onRunDiagnostic"
    />
    
    <UReportGenerateModal 
      v-model="openReportModal" 
      :loading="reportLoading" 
      @submit="onGenerateReport"
    />
    
    <UCreateSystemModal 
      v-model="openSystemModal" 
      :loading="systemLoading" 
      @submit="onCreateSystem"
    />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, useI18n, useSeoMeta, useHead } from '#imports'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
} from 'echarts/components'

use([
  CanvasRenderer,
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
])

const { t } = useI18n()

useSeoMeta({
  title: 'Dashboard | Hydraulic Diagnostic SaaS',
  description: 'Real-time hydraulic system monitoring dashboard with AI-powered diagnostics, predictive maintenance, and performance analytics',
  ogTitle: 'Dashboard | Hydraulic Diagnostic SaaS',
  ogDescription: 'Monitor your hydraulic systems in real-time with AI-powered insights',
  ogType: 'website',
  twitterCard: 'summary_large_image',
})

useHead({
  titleTemplate: (titleChunk) => {
    return titleChunk 
      ? `${titleChunk} | Hydraulic Diagnostic` 
      : 'Hydraulic Diagnostic SaaS'
  }
})

const openDiagnosticModal = ref(false)
const openReportModal = ref(false)
const openSystemModal = ref(false)
const diagnosticLoading = ref(false)
const reportLoading = ref(false)
const systemLoading = ref(false)
const performancePeriod = ref('24h')

const refreshData = () => {
  console.log('Refreshing dashboard data...')
}

const onRunDiagnostic = (data: any) => {
  console.log('Run diagnostic:', data)
  diagnosticLoading.value = true
  setTimeout(() => {
    diagnosticLoading.value = false
    openDiagnosticModal.value = false
  }, 2000)
}

const onGenerateReport = (data: any) => {
  console.log('Generate report:', data)
  reportLoading.value = true
  setTimeout(() => {
    reportLoading.value = false
    openReportModal.value = false
  }, 2000)
}

const onCreateSystem = (data: any) => {
  console.log('Create system:', data)
  systemLoading.value = true
  setTimeout(() => {
    systemLoading.value = false
    openSystemModal.value = false
  }, 2000)
}

const performanceChartOption = computed(() => ({
  tooltip: {
    trigger: 'axis',
    backgroundColor: '#1e293b',
    borderColor: '#334155',
    textStyle: { color: '#e2e8f0' },
  },
  grid: { 
    left: '5%', 
    right: '5%', 
    bottom: '10%', 
    top: '10%', 
    containLabel: true 
  },
  xAxis: {
    type: 'category',
    data: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
    axisLine: { lineStyle: { color: '#475569' } },
    axisLabel: { color: '#94a3b8' },
  },
  yAxis: {
    type: 'value',
    axisLine: { lineStyle: { color: '#475569' } },
    axisLabel: { color: '#94a3b8' },
    splitLine: { lineStyle: { color: '#334155' } },
  },
  series: [
    {
      data: [87, 92, 89, 95, 91, 97],
      type: 'line',
      smooth: true,
      lineStyle: { color: '#4f46e5', width: 3 },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(79, 70, 229, 0.3)' },
            { offset: 1, color: 'rgba(79, 70, 229, 0)' },
          ],
        },
      },
    },
  ],
}))

const aiChartOption = computed(() => ({
  tooltip: {
    trigger: 'axis',
    backgroundColor: '#1e293b',
    borderColor: '#334155',
    textStyle: { color: '#e2e8f0' },
  },
  grid: { 
    left: '5%', 
    right: '5%', 
    bottom: '10%', 
    top: '10%', 
    containLabel: true 
  },
  xAxis: {
    type: 'category',
    data: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    axisLine: { lineStyle: { color: '#475569' } },
    axisLabel: { color: '#94a3b8' },
  },
  yAxis: {
    type: 'value',
    axisLine: { lineStyle: { color: '#475569' } },
    axisLabel: { color: '#94a3b8' },
    splitLine: { lineStyle: { color: '#334155' } },
  },
  series: [
    {
      data: [12, 19, 15, 23, 18, 25, 21],
      type: 'line',
      smooth: true,
      lineStyle: { color: '#a855f7', width: 3 },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(168, 85, 247, 0.3)' },
            { offset: 1, color: 'rgba(168, 85, 247, 0)' },
          ],
        },
      },
    },
  ],
}))
</script>
