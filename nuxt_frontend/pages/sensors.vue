<template>
  <div class="space-y-8">
    <!-- Header -->
    <div class="u-flex-between">
      <div>
        <h1 class="u-h2">Sensor Data</h1>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          Real-time monitoring and analysis of hydraulic system sensors
        </p>
      </div>
      <div class="flex items-center gap-2">
        <button class="u-btn u-btn-secondary u-btn-md">
          <Icon name="heroicons:arrow-up-tray" class="w-4 h-4 mr-2" />
          Upload CSV
        </button>
        <button class="u-btn u-btn-primary u-btn-md">
          <Icon name="heroicons:arrow-path" class="w-4 h-4 mr-2" />
          Refresh Data
        </button>
      </div>
    </div>

    <!-- Live Stats -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Active Sensors</h3>
          <div class="u-metric-icon bg-green-100 dark:bg-green-900/30">
            <Icon name="heroicons:signal" class="w-5 h-5 text-green-600 dark:text-green-400" />
          </div>
        </div>
        <div class="u-metric-value">24</div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>All online</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Avg Pressure</h3>
          <div class="u-metric-icon bg-blue-100 dark:bg-blue-900/30">
            <Icon name="heroicons:chart-bar" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
        <div class="u-metric-value">147<span class="text-lg text-gray-500">PSI</span></div>
        <div class="u-metric-change text-gray-600 dark:text-gray-400 mt-2">
          <Icon name="heroicons:minus" class="w-4 h-4" />
          <span>Within normal</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Temperature</h3>
          <div class="u-metric-icon bg-orange-100 dark:bg-orange-900/30">
            <Icon name="heroicons:fire" class="w-5 h-5 text-orange-600 dark:text-orange-400" />
          </div>
        </div>
        <div class="u-metric-value">72°<span class="text-lg text-gray-500">C</span></div>
        <div class="u-metric-change u-metric-change-negative mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>+3°C elevated</span>
        </div>
      </div>

      <div class="u-metric-card">
        <div class="u-metric-header">
          <h3 class="u-metric-label">Flow Rate</h3>
          <div class="u-metric-icon bg-cyan-100 dark:bg-cyan-900/30">
            <Icon name="heroicons:arrow-right-circle" class="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
          </div>
        </div>
        <div class="u-metric-value">24<span class="text-lg text-gray-500">L/m</span></div>
        <div class="u-metric-change u-metric-change-positive mt-2">
          <Icon name="heroicons:arrow-trending-up" class="w-4 h-4" />
          <span>Optimal flow</span>
        </div>
      </div>
    </div>

    <!-- View Controls -->
    <div class="u-card p-6">
      <div class="u-flex-between">
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-2">
            <button
              @click="viewMode = 'table'"
              class="u-btn u-btn-sm"
              :class="viewMode === 'table' ? 'u-btn-primary' : 'u-btn-ghost'"
            >
              <Icon name="heroicons:table-cells" class="w-4 h-4 mr-1" />
              Table
            </button>
            <button
              @click="viewMode = 'chart'"
              class="u-btn u-btn-sm"
              :class="viewMode === 'chart' ? 'u-btn-primary' : 'u-btn-ghost'"
            >
              <Icon name="heroicons:chart-bar" class="w-4 h-4 mr-1" />
              Charts
            </button>
          </div>
          
          <div class="flex items-center gap-2">
            <select v-model="timeRange" class="u-input text-sm py-1 px-2 w-32">
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
            
            <select v-model="equipmentFilter" class="u-input text-sm py-1 px-2 w-32">
              <option value="all">All Equipment</option>
              <option value="hyd-001">HYD-001</option>
              <option value="hyd-002">HYD-002</option>
              <option value="hyd-003">HYD-003</option>
            </select>
          </div>
        </div>
        
        <button class="u-btn u-btn-secondary u-btn-sm">
          <Icon name="heroicons:arrow-down-tray" class="w-4 h-4 mr-1" />
          Export
        </button>
      </div>
    </div>

    <!-- Table View -->
    <div v-if="viewMode === 'table'" class="u-card">
      <div class="p-6 border-b border-gray-200 dark:border-gray-700">
        <h3 class="u-h4">Live Sensor Readings</h3>
        <p class="u-body text-gray-600 dark:text-gray-400 mt-1">
          Real-time and historical sensor data from all hydraulic systems
        </p>
      </div>
      
      <div class="overflow-x-auto">
        <table class="u-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Equipment</th>
              <th>Pressure</th>
              <th>Temperature</th>
              <th>Flow Rate</th>
              <th>Vibration</th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="reading in filteredSensorData" :key="reading.id">
              <td class="font-mono u-body-sm">{{ reading.timestamp }}</td>
              <td>
                <div class="flex items-center gap-2">
                  <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-gray-400" />
                  <span class="font-medium">{{ reading.equipment }}</span>
                </div>
              </td>
              <td>
                <div class="flex items-center gap-2">
                  <div class="w-2 h-2 rounded-full" :class="getPressureColor(reading.pressure)"></div>
                  <span>{{ reading.pressure }} PSI</span>
                </div>
              </td>
              <td>
                <div class="flex items-center gap-2">
                  <Icon name="heroicons:fire" class="w-4 h-4" :class="getTemperatureColor(reading.temperature)" />
                  <span>{{ reading.temperature }}°C</span>
                </div>
              </td>
              <td>{{ reading.flowRate }} L/min</td>
              <td>{{ reading.vibration }} mm/s</td>
              <td>
                <span class="u-badge" :class="getStatusBadgeClass(reading.status)">
                  <Icon :name="getStatusIcon(reading.status)" class="w-3 h-3" />
                  {{ reading.status }}
                </span>
              </td>
              <td>
                <div class="flex items-center gap-1">
                  <button class="u-btn u-btn-ghost u-btn-sm">
                    <Icon name="heroicons:chart-bar" class="w-4 h-4" />
                  </button>
                  <button class="u-btn u-btn-ghost u-btn-sm">
                    <Icon name="heroicons:cog-6-tooth" class="w-4 h-4" />
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Chart View -->
    <div v-else class="space-y-6">
      <div class="u-chart-wrapper">
        <div class="u-chart-header">
          <h3 class="u-chart-title">Pressure Trends</h3>
          <div class="u-chart-controls">
            <select class="u-input text-sm py-1 px-2 w-32">
              <option>Real-time</option>
              <option>Last Hour</option>
              <option>Last Day</option>
            </select>
          </div>
        </div>
        <div class="u-chart-container u-flex-center">
          <div class="text-center">
            <Icon name="heroicons:chart-bar" class="w-12 h-12 mx-auto text-gray-400 mb-2" />
            <p class="u-body text-gray-500 dark:text-gray-400">Pressure chart will be rendered here</p>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div class="u-chart-wrapper">
          <div class="u-chart-header">
            <h3 class="u-chart-title">Temperature</h3>
          </div>
          <div class="u-chart-container u-flex-center">
            <div class="text-center">
              <Icon name="heroicons:fire" class="w-12 h-12 mx-auto text-orange-400 mb-2" />
              <p class="u-body text-gray-500 dark:text-gray-400">Temperature chart</p>
            </div>
          </div>
        </div>

        <div class="u-chart-wrapper">
          <div class="u-chart-header">
            <h3 class="u-chart-title">Flow Rate</h3>
          </div>
          <div class="u-chart-container u-flex-center">
            <div class="text-center">
              <Icon name="heroicons:arrow-right-circle" class="w-12 h-12 mx-auto text-cyan-400 mb-2" />
              <p class="u-body text-gray-500 dark:text-gray-400">Flow rate chart</p>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Live Data Stream -->
    <div class="u-card p-6">
      <div class="u-flex-between mb-4">
        <h3 class="u-h4">Real-time Data Stream</h3>
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span class="u-body-sm text-green-600 dark:text-green-400">Live</span>
        </div>
      </div>
      
      <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
        <div class="u-flex-between">
          <div class="flex items-center gap-4">
            <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <div>
              <p class="font-medium text-gray-900 dark:text-white">Data Stream Active</p>
              <p class="u-body-sm text-gray-500 dark:text-gray-400">Receiving updates every 30 seconds</p>
            </div>
          </div>
          <div class="text-right">
            <p class="u-body-sm text-gray-500 dark:text-gray-400">Last update</p>
            <p class="font-mono text-sm font-medium text-gray-900 dark:text-white">{{ currentTime }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  title: 'Sensor Data',
  layout: 'default',
  middleware: ['auth']
})

interface SensorReading {
  id: number
  timestamp: string
  equipment: string
  pressure: number
  temperature: number
  flowRate: number
  vibration: number
  status: 'normal' | 'warning' | 'critical'
}

// State
const viewMode = ref('table')
const timeRange = ref('24h')
const equipmentFilter = ref('all')
const currentTime = ref('')

// Demo sensor data
const sensorData = ref<SensorReading[]>([
  {
    id: 1,
    timestamp: '2024-01-20 14:30:00',
    equipment: 'HYD-001',
    pressure: 145,
    temperature: 68,
    flowRate: 22,
    vibration: 2.1,
    status: 'normal'
  },
  {
    id: 2,
    timestamp: '2024-01-20 14:29:30',
    equipment: 'HYD-001', 
    pressure: 148,
    temperature: 71,
    flowRate: 24,
    vibration: 2.3,
    status: 'normal'
  },
  {
    id: 3,
    timestamp: '2024-01-20 14:29:00',
    equipment: 'HYD-002',
    pressure: 152,
    temperature: 74,
    flowRate: 26,
    vibration: 3.1,
    status: 'warning'
  },
  {
    id: 4,
    timestamp: '2024-01-20 14:28:30',
    equipment: 'HYD-001',
    pressure: 149,
    temperature: 73,
    flowRate: 25,
    vibration: 2.2,
    status: 'normal'
  },
  {
    id: 5,
    timestamp: '2024-01-20 14:28:00',
    equipment: 'HYD-003',
    pressure: 160,
    temperature: 78,
    flowRate: 28,
    vibration: 4.2,
    status: 'critical'
  }
])

// Computed
const filteredSensorData = computed(() => {
  return sensorData.value.filter(reading => {
    if (equipmentFilter.value === 'all') return true
    return reading.equipment.toLowerCase() === equipmentFilter.value
  })
})

// Methods
const getPressureColor = (pressure: number) => {
  if (pressure < 140) return 'bg-green-500'
  if (pressure < 155) return 'bg-yellow-500' 
  return 'bg-red-500'
}

const getTemperatureColor = (temperature: number) => {
  if (temperature < 70) return 'text-green-500'
  if (temperature < 75) return 'text-yellow-500'
  return 'text-red-500'
}

const getStatusBadgeClass = (status: string) => {
  const classes = {
    'normal': 'u-badge-success',
    'warning': 'u-badge-warning',
    'critical': 'u-badge-error'
  }
  return classes[status] || 'u-badge-info'
}

const getStatusIcon = (status: string) => {
  const icons = {
    'normal': 'heroicons:check-circle',
    'warning': 'heroicons:exclamation-triangle', 
    'critical': 'heroicons:x-circle'
  }
  return icons[status] || 'heroicons:question-mark-circle'
}

const updateTime = () => {
  currentTime.value = new Date().toLocaleTimeString('en-US', {
    hour12: false,
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

// Lifecycle
let timeInterval: NodeJS.Timeout

onMounted(() => {
  updateTime()
  timeInterval = setInterval(updateTime, 1000)
})

onUnmounted(() => {
  if (timeInterval) {
    clearInterval(timeInterval)
  }
})
</script>