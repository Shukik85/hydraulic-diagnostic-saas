<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="premium-heading-xl text-gray-900 dark:text-white">Dashboard</h1>
        <p class="mt-1 premium-body text-gray-600 dark:text-gray-400">Hydraulic Systems Monitoring & Diagnostics</p>
      </div>
      
      <div class="flex items-center space-x-3">
        <button class="premium-button-primary premium-button-md">
          <Icon name="heroicons:arrow-path" class="w-4 h-4 mr-2" />
          Refresh
        </button>
      </div>
    </div>
    
    <!-- KPI Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <MetricCard
        title="Active Systems"
        :value="metrics.activeSystems"
        trend="+2"
        icon="heroicons:cpu-chip"
        type="success"
      />
      
      <MetricCard
        title="Critical Alerts"
        :value="metrics.criticalAlerts"
        trend="-1"
        icon="heroicons:exclamation-triangle"
        type="error"
      />
      
      <MetricCard
        title="Avg Pressure"
        :value="metrics.avgPressure"
        unit="MPa"
        trend="+0.2"
        icon="heroicons:beaker"
        type="info"
      />
      
      <MetricCard
        title="Efficiency"
        :value="metrics.efficiency"
        unit="%"
        trend="+1.2"
        icon="heroicons:chart-bar-square"
        type="success"
      />
    </div>
    
    <!-- Charts Row -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <SensorChart
        title="Pressure Timeline"
        type="line"
        :data="pressureData"
        unit="MPa"
      />
      
      <SensorChart
        title="System Temperature"
        type="gauge"
        :data="temperatureData"
        unit="°C"
      />
    </div>
    
    <!-- System Status Table -->
    <div class="premium-card">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <h3 class="premium-heading-md text-gray-900 dark:text-white">System Status</h3>
      </div>
      
      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
          <thead class="bg-gray-50 dark:bg-gray-800">
            <tr>
              <th class="px-6 py-3 text-left premium-heading-xs text-gray-500 dark:text-gray-400">
                System
              </th>
              <th class="px-6 py-3 text-left premium-heading-xs text-gray-500 dark:text-gray-400">
                Status
              </th>
              <th class="px-6 py-3 text-left premium-heading-xs text-gray-500 dark:text-gray-400">
                Pressure
              </th>
              <th class="px-6 py-3 text-left premium-heading-xs text-gray-500 dark:text-gray-400">
                Temperature
              </th>
              <th class="px-6 py-3 text-left premium-heading-xs text-gray-500 dark:text-gray-400">
                Last Update
              </th>
            </tr>
          </thead>
          <tbody class="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
            <tr v-for="system in systemsList" :key="system.id" class="premium-transition hover:bg-gray-50 dark:hover:bg-gray-700">
              <td class="px-6 py-4 whitespace-nowrap">
                <div class="flex items-center">
                  <Icon name="heroicons:cog-6-tooth" class="w-5 h-5 text-gray-400 mr-3" />
                  <div class="premium-body font-medium text-gray-900 dark:text-white">{{ system.name }}</div>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span 
                  class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                  :class="getStatusClass(system.status)"
                >
                  {{ system.status }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap premium-body text-gray-900 dark:text-white">
                {{ system.pressure }} MPa
              </td>
              <td class="px-6 py-4 whitespace-nowrap premium-body text-gray-900 dark:text-white">
                {{ system.temperature }}°C
              </td>
              <td class="px-6 py-4 whitespace-nowrap premium-body text-gray-500 dark:text-gray-400">
                {{ formatTime(system.lastUpdate) }}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// Demo data - заменится на API данные
const metrics = ref({
  activeSystems: 12,
  criticalAlerts: 3,
  avgPressure: 2.4,
  efficiency: 94.2
})

// Demo chart data
const pressureData = ref([
  { time: '00:00', value: 2.1 },
  { time: '04:00', value: 2.3 },
  { time: '08:00', value: 2.5 },
  { time: '12:00', value: 2.4 },
  { time: '16:00', value: 2.6 },
  { time: '20:00', value: 2.2 },
  { time: '24:00', value: 2.4 }
])

const temperatureData = ref([
  { value: 65, max: 100 }
])

const systemsList = ref([
  { 
    id: 1, 
    name: 'Pump System A', 
    status: 'online', 
    pressure: 2.4, 
    temperature: 65,
    lastUpdate: new Date(Date.now() - 5 * 60 * 1000)
  },
  { 
    id: 2, 
    name: 'Hydraulic Unit B', 
    status: 'warning', 
    pressure: 1.8, 
    temperature: 78,
    lastUpdate: new Date(Date.now() - 15 * 60 * 1000)
  },
  { 
    id: 3, 
    name: 'Cooling System', 
    status: 'online', 
    pressure: 2.1, 
    temperature: 45,
    lastUpdate: new Date(Date.now() - 2 * 60 * 1000)
  }
])

const getStatusClass = (status: string) => {
  const classes = {
    'online': 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400',
    'warning': 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400',
    'error': 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400',
    'offline': 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  }
  return classes[status] || classes['offline']
}

const formatTime = (date: Date) => {
  return new Intl.RelativeTimeFormat('ru', { numeric: 'auto' }).format(
    Math.round((date.getTime() - Date.now()) / (1000 * 60)), 'minute'
  )
}

definePageMeta({
  title: 'Dashboard',
  layout: 'default'
})
</script>