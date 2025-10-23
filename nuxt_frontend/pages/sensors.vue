<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Sensor Data</h1>
        <p class="text-muted-foreground">Upload and analyze sensor data from your hydraulic systems</p>
      </div>
      <div class="flex items-center gap-2">
        <UiButton variant="outline">
          <Icon name="lucide:upload" class="mr-2 h-4 w-4" />
          Upload CSV
        </UiButton>
        <UiButton>
          <Icon name="lucide:refresh-cw" class="mr-2 h-4 w-4" />
          Refresh Data
        </UiButton>
      </div>
    </div>

    <!-- Data Controls -->
    <UiCard>
      <UiCardContent class="pt-6">
        <div class="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div class="flex items-center gap-4">
            <div class="flex items-center gap-2">
              <UiButton
                variant="outline"
                size="sm"
                @click="viewMode = 'table'"
                :class="{ 'bg-muted': viewMode === 'table' }"
              >
                <Icon name="lucide:table" class="h-4 w-4" />
              </UiButton>
              <UiButton
                variant="outline"
                size="sm"
                @click="viewMode = 'chart'"
                :class="{ 'bg-muted': viewMode === 'chart' }"
              >
                <Icon name="lucide:bar-chart-3" class="h-4 w-4" />
              </UiButton>
            </div>
            <UiSelect v-model="timeRange">
              <UiSelectItem value="1h">Last Hour</UiSelectItem>
              <UiSelectItem value="24h">Last 24 Hours</UiSelectItem>
              <UiSelectItem value="7d">Last 7 Days</UiSelectItem>
              <UiSelectItem value="30d">Last 30 Days</UiSelectItem>
            </UiSelect>
            <UiSelect v-model="equipmentFilter">
              <UiSelectItem value="all">All Equipment</UiSelectItem>
              <UiSelectItem value="hyd-001">HYD-001</UiSelectItem>
              <UiSelectItem value="hyd-002">HYD-002</UiSelectItem>
              <UiSelectItem value="hyd-003">HYD-003</UiSelectItem>
            </UiSelect>
          </div>
          <div class="flex items-center gap-2">
            <UiButton variant="outline" size="sm">
              <Icon name="lucide:download" class="mr-2 h-4 w-4" />
              Export
            </UiButton>
          </div>
        </div>
      </UiCardContent>
    </UiCard>

    <!-- Table View -->
    <UiCard v-if="viewMode === 'table'">
      <UiCardHeader>
        <UiCardTitle>Sensor Readings</UiCardTitle>
        <UiCardDescription>Real-time and historical sensor data</UiCardDescription>
      </UiCardHeader>
      <UiCardContent>
        <div class="rounded-md border">
          <table class="w-full">
            <thead>
              <tr class="border-b">
                <th class="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Timestamp</th>
                <th class="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Equipment</th>
                <th class="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Pressure (PSI)</th>
                <th class="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Temperature (°C)</th>
                <th class="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Flow Rate (L/min)</th>
                <th class="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Vibration (mm/s)</th>
                <th class="h-12 px-4 text-left align-middle font-medium text-muted-foreground">Status</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="reading in sensorData" :key="reading.id" class="border-b">
                <td class="p-4 align-middle">{{ reading.timestamp }}</td>
                <td class="p-4 align-middle font-medium">{{ reading.equipment }}</td>
                <td class="p-4 align-middle">{{ reading.pressure }}</td>
                <td class="p-4 align-middle">{{ reading.temperature }}</td>
                <td class="p-4 align-middle">{{ reading.flowRate }}</td>
                <td class="p-4 align-middle">{{ reading.vibration }}</td>
                <td class="p-4 align-middle">
                  <span
                    :class="`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                      reading.status === 'normal' ? 'bg-status-success/10 text-status-success' :
                      reading.status === 'warning' ? 'bg-status-warning/10 text-status-warning' :
                      'bg-status-error/10 text-status-error'
                    }`"
                  >
                    {{ reading.status }}
                  </span>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </UiCardContent>
    </UiCard>

    <!-- Chart View -->
    <div v-else class="grid gap-6">
      <UiCard>
        <UiCardHeader>
          <UiCardTitle>Pressure Trends</UiCardTitle>
          <UiCardDescription>Pressure readings over time</UiCardDescription>
        </UiCardHeader>
        <UiCardContent>
          <div class="h-[300px] w-full bg-muted/20 rounded-md flex items-center justify-center">
            <div class="text-center">
              <Icon name="lucide:trending-up" class="mx-auto h-12 w-12 text-muted-foreground mb-2" />
              <p class="text-sm text-muted-foreground">Pressure chart will be rendered here</p>
            </div>
          </div>
        </UiCardContent>
      </UiCard>

      <UiCard>
        <UiCardHeader>
          <UiCardTitle>Temperature & Flow Rate</UiCardTitle>
          <UiCardDescription>Temperature and flow rate correlation</UiCardDescription>
        </UiCardHeader>
        <UiCardContent>
          <div class="h-[300px] w-full bg-muted/20 rounded-md flex items-center justify-center">
            <div class="text-center">
              <Icon name="lucide:activity" class="mx-auto h-12 w-12 text-muted-foreground mb-2" />
              <p class="text-sm text-muted-foreground">Temperature & flow chart will be rendered here</p>
            </div>
          </div>
        </UiCardContent>
      </UiCard>
    </div>

    <!-- Real-time Updates -->
    <UiCard>
      <UiCardHeader>
        <UiCardTitle>Real-time Updates</UiCardTitle>
        <UiCardDescription>Live sensor data stream</UiCardDescription>
      </UiCardHeader>
      <UiCardContent>
        <div class="flex items-center justify-between p-4 bg-muted/20 rounded-lg">
          <div class="flex items-center space-x-4">
            <div class="h-3 w-3 bg-status-success rounded-full animate-pulse"></div>
            <div>
              <p class="font-medium">Live Data Stream Active</p>
              <p class="text-sm text-muted-foreground">Receiving updates every 30 seconds</p>
            </div>
          </div>
          <div class="flex items-center space-x-2">
            <span class="text-sm text-muted-foreground">Last update:</span>
            <span class="text-sm font-medium">{{ currentTime }}</span>
          </div>
        </div>
      </UiCardContent>
    </UiCard>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'

const viewMode = ref('table')
const timeRange = ref('24h')
const equipmentFilter = ref('all')
const currentTime = ref('')

const sensorData = ref([
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
  }
])

const updateTime = () => {
  currentTime.value = new Date().toLocaleTimeString()
}

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
