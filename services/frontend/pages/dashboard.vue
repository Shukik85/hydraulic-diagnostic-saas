<template>
  <div class="space-y-10">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-white">{{ t('dashboard.title') }}</h1>
        <p class="text-steel-shine mt-1">{{ t('dashboard.subtitle') }}</p>
      </div>
      <div class="flex items-center gap-3">
        <div class="flex gap-2 items-center px-4 py-2 rounded-lg gradient-metal shadow-metal">
          <Icon name="heroicons:clock" class="w-4 h-4 text-steel-shine" />
          <span class="text-sm text-steel-shine">{{ t('dashboard.liveStatus') }}</span>
          <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></span>
        </div>
        <button class="btn-metal px-4 py-2">
          <Icon name="heroicons:arrow-path" class="w-4 h-4 mr-2" />
          {{ t('dashboard.refreshBtn') }}
        </button>
      </div>
    </div>

    <!-- KPI Metallic Grid -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <div class="card-metal">
        <div class="flex items-center justify-between mb-1">
          <h3 class="text-base font-semibold text-steel-shine">{{ t('dashboard.kpi.activeSystems') }}</h3>
          <div class="bg-gradient-to-tr from-blue-500/40 to-steel-shine rounded p-2">
            <Icon name="heroicons:server-stack" class="w-5 h-5 text-blue-300" />
          </div>
        </div>
        <div class="text-3xl font-bold text-white mb-2">127</div>
        <div class="flex items-center text-xs text-success-500 gap-1">
          <Icon name="heroicons:arrow-trending-up" class="w-3 h-3" />
          <span>+5 {{ t('dashboard.kpi.fromYesterday') }}</span>
        </div>
      </div>
      <div class="card-metal">
        <div class="flex items-center justify-between mb-1">
          <h3 class="text-base font-semibold text-steel-shine">{{ t('dashboard.kpi.systemHealth') }}</h3>
          <div class="bg-gradient-to-tr from-green-500/40 to-steel-shine rounded p-2">
            <Icon name="heroicons:heart" class="w-5 h-5 text-green-300" />
          </div>
        </div>
        <div class="text-3xl font-bold text-white mb-2">99.9%</div>
        <div class="flex items-center text-xs text-success-500 gap-1">
          <Icon name="heroicons:arrow-trending-up" class="w-3 h-3" />
          <span>{{ t('dashboard.kpi.uptimeExcellence') }}</span>
        </div>
      </div>
      <div class="card-metal">
        <div class="flex items-center justify-between mb-1">
          <h3 class="text-base font-semibold text-steel-shine">{{ t('dashboard.kpi.preventedFailures') }}</h3>
          <div class="bg-gradient-to-tr from-purple-500/40 to-steel-shine rounded p-2">
            <Icon name="heroicons:shield-check" class="w-5 h-5 text-purple-300" />
          </div>
        </div>
        <div class="text-3xl font-bold text-white mb-2">23</div>
        <div class="flex items-center text-xs text-success-500 gap-1">
          <Icon name="heroicons:arrow-trending-up" class="w-3 h-3" />
          <span>{{ t('dashboard.kpi.aiPredictions') }}</span>
        </div>
      </div>
      <div class="card-metal">
        <div class="flex items-center justify-between mb-1">
          <h3 class="text-base font-semibold text-steel-shine">{{ t('dashboard.kpi.costSavings') }}</h3>
          <div class="bg-gradient-to-tr from-orange-500/40 to-steel-shine rounded p-2">
            <Icon name="heroicons:currency-dollar" class="w-5 h-5 text-orange-300" />
          </div>
        </div>
        <div class="text-3xl font-bold text-white mb-2">89%</div>
        <div class="flex items-center text-xs text-success-500 gap-1">
          <Icon name="heroicons:arrow-trending-up" class="w-3 h-3" />
          <span>{{ t('dashboard.kpi.vsPreviousYear') }}</span>
        </div>
      </div>
    </div>

    <!-- Charts Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div class="card-metal p-0">
        <div class="flex justify-between items-center p-6 border-b border-steel-light">
          <h3 class="font-semibold text-white">{{ t('dashboard.charts.systemPerformance') }}</h3>
          <select class="input-metal text-sm py-1 px-2 w-32">
            <option>{{ t('dashboard.charts.last24h') }}</option>
            <option>{{ t('dashboard.charts.last7d') }}</option>
            <option>{{ t('dashboard.charts.last30d') }}</option>
          </select>
        </div>
        <div class="u-chart-container p-6">
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

      <div class="card-metal p-0">
        <div class="flex justify-between items-center p-6 border-b border-steel-light">
          <h3 class="font-semibold text-white">{{ t('dashboard.charts.aiPredictions') }}</h3>
          <span class="badge-status badge-info">
            <Icon name="heroicons:sparkles" class="w-4 h-4" />
            {{ t('dashboard.charts.mlActive') }}
          </span>
        </div>
        <div class="u-chart-container p-6">
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

    <!-- Recent Activity & Quick Actions Grid -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div class="card-metal p-6">
        <h3 class="text-lg font-semibold text-white mb-6">{{ t('dashboard.quickActions.title') }}</h3>
        <div class="space-y-3">
          <button @click="openDiagnosticModal = true" class="w-full flex items-center gap-3 p-3 rounded-lg bg-primary-600/10 hover:bg-primary-600/20 text-primary-200 shadow-metal transition text-left">
            <Icon name="heroicons:play" class="w-5 h-5" />
            <span class="text-sm font-medium">{{ t('dashboard.quickActions.runDiagnostics') }}</span>
          </button>
          <button @click="openReportModal = true" class="w-full flex items-center gap-3 p-3 rounded-lg bg-success-500/10 hover:bg-success-500/20 text-success-500 shadow-metal transition text-left">
            <Icon name="heroicons:document-text" class="w-5 h-5" />
            <span class="text-sm font-medium">{{ t('dashboard.quickActions.generateReport') }}</span>
          </button>
          <button @click="openSystemModal = true" class="w-full flex items-center gap-3 p-3 rounded-lg bg-purple-700/10 hover:bg-purple-700/20 text-purple-400 shadow-metal transition text-left">
            <Icon name="heroicons:plus" class="w-5 h-5" />
            <span class="text-sm font-medium">{{ t('dashboard.quickActions.addSystem') }}</span>
          </button>
        </div>
      </div>
      <div class="card-metal p-6">
        <h3 class="text-lg font-semibold text-white mb-6">{{ t('dashboard.recentEvents.title') }}</h3>
        <div class="space-y-3">
          <div class="flex items-center gap-3 p-3 rounded-lg bg-metal-medium/70">
            <div class="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-white">
                {{ t('dashboard.recentEvents.systemNormal', ['127']) }}
              </p>
              <p class="text-xs text-steel-shine">{{ t('dashboard.recentEvents.minutesAgo', ['2']) }}</p>
            </div>
          </div>
          <div class="flex items-center gap-3 p-3 rounded-lg bg-status-warning/10">
            <div class="w-2 h-2 bg-yellow-400 rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-white">
                {{ t('dashboard.recentEvents.anomalyDetected', ['89']) }}
              </p>
              <p class="text-xs text-steel-shine">{{ t('dashboard.recentEvents.minutesAgo', ['15']) }}</p>
            </div>
          </div>
          <div class="flex items-center gap-3 p-3 rounded-lg bg-status-info/10">
            <div class="w-2 h-2 bg-blue-400 rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-white">
                {{ t('dashboard.recentEvents.diagnosticsCompleted', ['45']) }}
              </p>
              <p class="text-xs text-steel-shine">{{ t('dashboard.recentEvents.hourAgo', ['1']) }}</p>
            </div>
          </div>
        </div>
      </div>
      <div class="card-metal p-6">
        <h3 class="text-lg font-semibold text-white mb-6">{{ t('dashboard.aiStatus.title') }}</h3>
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.diagnosticModel') }}</span>
            <span class="badge-status badge-success">{{ t('dashboard.aiStatus.active') }}</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.predictionAccuracy') }}</span>
            <span class="text-sm font-semibold text-primary-400">97.3%</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.modelTraining') }}</span>
            <span class="badge-status badge-info">{{ t('dashboard.aiStatus.complete') }}</span>
          </div>
          <div class="border-t border-steel-light my-4"></div>
          <div class="flex items-center justify-between">
            <span class="text-steel-shine">{{ t('dashboard.aiStatus.lastUpdate') }}</span>
            <span class="text-xs text-steel-shine">{{ t('dashboard.aiStatus.minAgo', ['5']) }}</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Modal Components remain unchanged -->
    <URunDiagnosticModal  v-model="openDiagnosticModal" :loading="diagnosticLoading" @submit="onRunDiagnostic" @cancel="onCancelDiagnostic" />
    <UReportGenerateModal v-model="openReportModal" :loading="reportLoading" @submit="onGenerateReport" @cancel="onCancelReport" />
    <UCreateSystemModal   v-model="openSystemModal" :loading="systemLoading" @submit="onCreateSystem" @cancel="onCancelSystem" />
  </div>
</template>

<script setup lang="ts">
// ...оставить нынешнюю бизнес-логику...
</script>
