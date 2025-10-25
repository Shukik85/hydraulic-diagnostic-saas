<script setup lang='ts'>
import DashboardCharts from '~/components/dashboard/DashboardCharts.client.vue'
import Sparklines from '~/components/dashboard/Sparklines.client.vue'

const source = ref<'btc' | 'eth'>('btc')
const { data, refresh, pending } = await useFetch('/api/demo/hydraulic-metrics', {
  query: { source },
  key: () => `demo-hydraulics-${source.value}`,
  dedupe: 'defer',
  initialCache: false
})

const temp = computed(() => data.value?.sparklines?.temperature || [])
const pressure = computed(() => data.value?.sparklines?.pressure || [])
const flow = computed(() => data.value?.sparklines?.flow_rate || [])
const vibration = computed(() => data.value?.sparklines?.vibration || [])
const thresholds = computed(() => data.value?.thresholds || {})
const weekStats = computed(() => data.value?.aggregates?.week_stats || {})

// Auto-refresh every 12 seconds with noise injection for demo
let refreshTimer: NodeJS.Timeout | null = null

onMounted(() => {
  refreshTimer = setInterval(() => {
    // Add slight noise to demonstrate live updates
    if (data.value?.sparklines) {
      const noise = () => (Math.random() - 0.5) * 0.3
      const current = data.value.sparklines
      
      // Inject small variations to make updates visible
      current.temperature = current.temperature.map(v => Math.round((v + noise()) * 10) / 10)
      current.pressure = current.pressure.map(v => Math.round((v + noise()) * 10) / 10)
      current.flow_rate = current.flow_rate.map(v => Math.round((v + noise()) * 10) / 10)
      current.vibration = current.vibration.map(v => Math.round((v + noise() * 0.1) * 100) / 100)
    }
  }, 12000)
})

onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
})

watch(source, () => refresh())

const zoneColor = (metric: string, value: number) => {
  const t = thresholds.value?.[metric]
  if (!t || value == null) return 'text-gray-600 dark:text-gray-300'
  if (metric === 'pressure') {
    return value >= t.green ? 'text-green-600 dark:text-green-400' : (value <= t.red ? 'text-red-600 dark:text-red-400' : 'text-yellow-600 dark:text-yellow-400')
  }
  return value <= t.green ? 'text-green-600 dark:text-green-400' : (value >= t.red ? 'text-red-600 dark:text-red-400' : 'text-yellow-600 dark:text-yellow-400')
}
</script>

<template>
  <div class="space-y-6">
    <!-- Source Switcher -->
    <div class="premium-card p-4 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <span class="text-sm font-medium text-gray-700 dark:text-gray-300">📈 Источник данных:</span>
        <div class="flex rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700 shadow-sm">
          <button
            :class="[
              'px-4 py-2 text-sm font-medium transition-all',
              source === 'btc' 
                ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-sm' 
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-blue-50 dark:hover:bg-blue-900/30'
            ]"
            @click="source = 'btc'"
          >
            Bitcoin (Нестабильная)
          </button>
          <button
            :class="[
              'px-4 py-2 text-sm font-medium transition-all',
              source === 'eth' 
                ? 'bg-gradient-to-r from-purple-600 to-purple-700 text-white shadow-sm' 
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-purple-50 dark:hover:bg-purple-900/30'
            ]"
            @click="source = 'eth'"
          >
            Ethereum (Экстремальная)
          </button>
        </div>
      </div>
      <div v-if="pending" class="flex items-center space-x-2 text-sm text-blue-600 dark:text-blue-400">
        <div class="w-4 h-4 border-2 border-blue-600 dark:border-blue-400 border-t-transparent rounded-full animate-spin"></div>
        <span>Загрузка...</span>
      </div>
    </div>

    <!-- Sparklines -->
    <Sparklines :temp="temp" :pressure="pressure" :flow="flow" :vibration="vibration" />

    <!-- Weekly Aggregates with Zone Colors -->
    <div class="premium-card p-4">
      <h4 class="text-sm font-semibold text-gray-800 dark:text-gray-200 mb-3">🔍 Агрегаты за неделю</h4>
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
        <div class="text-center p-3 bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-lg border border-red-100 dark:border-red-800">
          <div class="text-xs text-gray-500 dark:text-gray-400 mb-2">Температура (°C)</div>
          <div class="space-y-1">
            <div :class="zoneColor('temperature', weekStats.temperature?.avg)" class="font-bold text-lg">
              ср: {{ weekStats.temperature?.avg }}
            </div>
            <div class="flex justify-between text-xs text-gray-600 dark:text-gray-300">
              <span>мин: {{ weekStats.temperature?.min }}</span>
              <span>макс: {{ weekStats.temperature?.max }}</span>
            </div>
          </div>
        </div>
        
        <div class="text-center p-3 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-100 dark:border-blue-800">
          <div class="text-xs text-gray-500 dark:text-gray-400 mb-2">Давление (бар)</div>
          <div class="space-y-1">
            <div :class="zoneColor('pressure', weekStats.pressure?.avg)" class="font-bold text-lg">
              ср: {{ weekStats.pressure?.avg }}
            </div>
            <div class="flex justify-between text-xs text-gray-600 dark:text-gray-300">
              <span>мин: {{ weekStats.pressure?.min }}</span>
              <span>макс: {{ weekStats.pressure?.max }}</span>
            </div>
          </div>
        </div>
        
        <div class="text-center p-3 bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-100 dark:border-green-800">
          <div class="text-xs text-gray-500 dark:text-gray-400 mb-2">Расход (л/мин)</div>
          <div class="space-y-1">
            <div :class="zoneColor('flow_rate', weekStats.flow_rate?.avg)" class="font-bold text-lg">
              ср: {{ weekStats.flow_rate?.avg }}
            </div>
            <div class="flex justify-between text-xs text-gray-600 dark:text-gray-300">
              <span>мин: {{ weekStats.flow_rate?.min }}</span>
              <span>макс: {{ weekStats.flow_rate?.max }}</span>
            </div>
          </div>
        </div>
        
        <div class="text-center p-3 bg-gradient-to-br from-purple-50 to-violet-50 dark:from-purple-900/20 dark:to-violet-900/20 rounded-lg border border-purple-100 dark:border-purple-800">
          <div class="text-xs text-gray-500 dark:text-gray-400 mb-2">Вибрация (мм/с)</div>
          <div class="space-y-1">
            <div :class="zoneColor('vibration', weekStats.vibration?.avg)" class="font-bold text-lg">
              ср: {{ weekStats.vibration?.avg }}
            </div>
            <div class="flex justify-between text-xs text-gray-600 dark:text-gray-300">
              <span>мин: {{ weekStats.vibration?.min }}</span>
              <span>макс: {{ weekStats.vibration?.max }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Main Charts -->
    <DashboardCharts />
  </div>
</template>