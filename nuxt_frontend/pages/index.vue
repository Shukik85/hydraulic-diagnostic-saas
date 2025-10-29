<template>
  <div class="space-y-6">
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p class="mt-1 text-gray-600">Hydraulic Systems Monitoring & Diagnostics</p>
      </div>

      <div class="flex items-center space-x-3">
        <button
          class="inline-flex items-center px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700 transition-colors">
          <Icon name="heroicons:arrow-path" class="w-4 h-4 mr-2" />
          Refresh
        </button>
      </div>
    </div>

    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <MetricCard title="Active Systems" :value="metrics.activeSystems" trend="+2" icon="heroicons:cpu-chip"
        type="success" />

      <MetricCard title="Critical Alerts" :value="metrics.criticalAlerts" trend="-1"
        icon="heroicons:exclamation-triangle" type="error" />

      <MetricCard title="Avg Pressure" :value="metrics.avgPressure" unit="MPa" trend="+0.2" icon="heroicons:beaker"
        type="info" />

      <MetricCard title="Efficiency" :value="metrics.efficiency" unit="%" trend="+1.2" icon="heroicons:chart-bar-square"
        type="success" />
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <SensorChart title="Pressure Timeline" type="line" :data="pressureData" unit="MPa" />

      <SensorChart title="System Temperature" type="gauge" :data="temperatureData" unit="°C" />
    </div>

    <div class="bg-white rounded-lg shadow-sm border border-gray-200">
      <div class="px-6 py-4 border-b border-gray-200">
        <h3 class="text-lg font-semibold text-gray-900">System Status</h3>
      </div>

      <div class="overflow-x-auto">
        <table class="min-w-full divide-y divide-gray-200">
          <thead class="bg-gray-50">
            <tr>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">System</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Pressure</th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Temperature
              </th>
              <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Last Update
              </th>
            </tr>
          </thead>
          <tbody class="bg-white divide-y divide-gray-200">
            <tr v-for="system in systemsList" :key="system.id">
              <td class="px-6 py-4 whitespace-nowrap">
                <div class="flex items-center">
                  <Icon name="heroicons:cog-6-tooth" class="w-5 h-5 text-gray-400 mr-3" />
                  <div class="text-sm font-medium text-gray-900">{{ system.name }}</div>
                </div>
              </td>
              <td class="px-6 py-4 whitespace-nowrap">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                  :class="getStatusClass(system.status)">
                  {{ system.status }}
                </span>
              </td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{{ system.pressure }} MPa</td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">{{ system.temperature }}°C</td>
              <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{{ formatTime(system.lastUpdate) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const metrics = ref({
  activeSystems: 12,
  criticalAlerts: 3,
  avgPressure: 2.4,
  efficiency: 94.2
})

const pressureData = ref([
  { time: '00:00', value: 2.1 },
  { time: '04:00', value: 2.3 },
  { time: '08:00', value: 2.5 },
  { time: '12:00', value: 2.4 },
  { time: '16:00', value: 2.6 },
  { time: '20:00', value: 2.2 },
  { time: '24:00', value: 2.4 }
])

const temperatureData = ref([{ value: 65, max: 100 }])

const systemsList = ref([
  { id: 1, name: 'Pump System A', status: 'online', pressure: 2.4, temperature: 65, lastUpdate: new Date(Date.now() - 5 * 60 * 1000) },
  { id: 2, name: 'Hydraulic Unit B', status: 'warning', pressure: 1.8, temperature: 78, lastUpdate: new Date(Date.now() - 15 * 60 * 1000) },
  { id: 3, name: 'Cooling System', status: 'online', pressure: 2.1, temperature: 45, lastUpdate: new Date(Date.now() - 2 * 60 * 1000) }
])

const getStatusClass = (status: string) => {
  const classes = {
    'online': 'bg-green-100 text-green-800',
    'warning': 'bg-yellow-100 text-yellow-800',
    'error': 'bg-red-100 text-red-800',
    'offline': 'bg-gray-100 text-gray-800'
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