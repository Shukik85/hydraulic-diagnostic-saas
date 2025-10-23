<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div class="flex items-center space-x-4">
        <UiButton variant="ghost" size="icon" @click="$router.back()">
          <Icon name="lucide:arrow-left" class="h-4 w-4" />
        </UiButton>
        <div>
          <h1 class="text-3xl font-bold tracking-tight">{{ equipment.name }}</h1>
          <p class="text-muted-foreground">{{ equipment.type }} • {{ equipment.location }}</p>
        </div>
      </div>
      <div class="flex items-center gap-2">
        <UiButton variant="outline">
          <Icon name="lucide:settings" class="mr-2 h-4 w-4" />
          Configure
        </UiButton>
        <UiButton variant="outline">
          <Icon name="lucide:download" class="mr-2 h-4 w-4" />
          Export
        </UiButton>
        <UiButton>
          <Icon name="lucide:play" class="mr-2 h-4 w-4" />
          Run Diagnostics
        </UiButton>
      </div>
    </div>

    <!-- Status Overview -->
    <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">Status</UiCardTitle>
          <div :class="`h-3 w-3 rounded-full ${getStatusColor(equipment.status)}`"></div>
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold capitalize">{{ equipment.status }}</div>
          <p class="text-xs text-muted-foreground">Current operational status</p>
        </UiCardContent>
      </UiCard>

      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">Health Score</UiCardTitle>
          <Icon name="lucide:activity" class="h-4 w-4 text-muted-foreground" />
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold">{{ equipment.health }}%</div>
          <p class="text-xs text-muted-foreground">Overall system health</p>
        </UiCardContent>
      </UiCard>

      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">Uptime</UiCardTitle>
          <Icon name="lucide:clock" class="h-4 w-4 text-muted-foreground" />
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold">99.2%</div>
          <p class="text-xs text-muted-foreground">Last 30 days</p>
        </UiCardContent>
      </UiCard>

      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">Last Service</UiCardTitle>
          <Icon name="lucide:calendar" class="h-4 w-4 text-muted-foreground" />
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold">2 weeks</div>
          <p class="text-xs text-muted-foreground">Next service due in 2 weeks</p>
        </UiCardContent>
      </UiCard>
    </div>

    <!-- Tabs -->
    <UiTabs v-model="activeTab" class="w-full">
      <UiTabsList class="grid w-full grid-cols-5">
        <UiTabsTrigger value="overview">Overview</UiTabsTrigger>
        <UiTabsTrigger value="sensors">Sensors</UiTabsTrigger>
        <UiTabsTrigger value="history">History</UiTabsTrigger>
        <UiTabsTrigger value="diagnostics">Diagnostics</UiTabsTrigger>
        <UiTabsTrigger value="docs">Documentation</UiTabsTrigger>
      </UiTabsList>

      <UiTabsContent value="overview" class="space-y-6">
        <!-- Real-time Metrics -->
        <div class="grid gap-4 md:grid-cols-2">
          <UiCard>
            <UiCardHeader>
              <UiCardTitle>Current Readings</UiCardTitle>
              <UiCardDescription>Real-time sensor data</UiCardDescription>
            </UiCardHeader>
            <UiCardContent>
              <div class="space-y-4">
                <div class="flex items-center justify-between">
                  <span class="text-sm text-muted-foreground">Pressure</span>
                  <span class="text-sm font-medium">145 PSI</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-muted-foreground">Temperature</span>
                  <span class="text-sm font-medium">68°C</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-muted-foreground">Flow Rate</span>
                  <span class="text-sm font-medium">22 L/min</span>
                </div>
                <div class="flex items-center justify-between">
                  <span class="text-sm text-muted-foreground">Vibration</span>
                  <span class="text-sm font-medium">2.1 mm/s</span>
                </div>
              </div>
            </UiCardContent>
          </UiCard>

          <UiCard>
            <UiCardHeader>
              <UiCardTitle>Performance Chart</UiCardTitle>
              <UiCardDescription>24-hour trend</UiCardDescription>
            </UiCardHeader>
            <UiCardContent>
              <div class="h-[200px] w-full bg-muted/20 rounded-md flex items-center justify-center">
                <div class="text-center">
                  <Icon name="lucide:trending-up" class="mx-auto h-8 w-8 text-muted-foreground mb-2" />
                  <p class="text-sm text-muted-foreground">Performance chart</p>
                </div>
              </div>
            </UiCardContent>
          </UiCard>
        </div>

        <!-- Recent Alerts -->
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Recent Alerts</UiCardTitle>
            <UiCardDescription>Latest system notifications</UiCardDescription>
          </UiCardHeader>
          <UiCardContent>
            <div class="space-y-3">
              <div class="flex items-start space-x-3">
                <div class="h-2 w-2 bg-status-warning rounded-full mt-2"></div>
                <div class="flex-1">
                  <p class="text-sm font-medium">Pressure fluctuation detected</p>
                  <p class="text-xs text-muted-foreground">15 minutes ago • Threshold: ±5 PSI</p>
                </div>
              </div>
              <div class="flex items-start space-x-3">
                <div class="h-2 w-2 bg-status-info rounded-full mt-2"></div>
                <div class="flex-1">
                  <p class="text-sm font-medium">Maintenance reminder</p>
                  <p class="text-xs text-muted-foreground">2 hours ago • Filter replacement due</p>
                </div>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <UiTabsContent value="sensors" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Sensor Configuration</UiCardTitle>
            <UiCardDescription>Connected sensors and their settings</UiCardDescription>
          </UiCardHeader>
          <UiCardContent>
            <div class="space-y-4">
              <div class="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <p class="font-medium">Pressure Sensor</p>
                  <p class="text-sm text-muted-foreground">Model: PT-100 • Range: 0-500 PSI</p>
                </div>
                <div class="flex items-center space-x-2">
                  <div class="h-2 w-2 bg-status-success rounded-full"></div>
                  <span class="text-sm text-muted-foreground">Online</span>
                </div>
              </div>

              <div class="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <p class="font-medium">Temperature Sensor</p>
                  <p class="text-sm text-muted-foreground">Model: TC-200 • Range: -20°C to 150°C</p>
                </div>
                <div class="flex items-center space-x-2">
                  <div class="h-2 w-2 bg-status-success rounded-full"></div>
                  <span class="text-sm text-muted-foreground">Online</span>
                </div>
              </div>

              <div class="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <p class="font-medium">Flow Meter</p>
                  <p class="text-sm text-muted-foreground">Model: FM-300 • Range: 0-100 L/min</p>
                </div>
                <div class="flex items-center space-x-2">
                  <div class="h-2 w-2 bg-status-warning rounded-full"></div>
                  <span class="text-sm text-muted-foreground">Warning</span>
                </div>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <UiTabsContent value="history" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Maintenance History</UiCardTitle>
            <UiCardDescription>Service records and maintenance activities</UiCardDescription>
          </UiCardHeader>
          <UiCardContent>
            <div class="space-y-4">
              <div class="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <p class="font-medium">Filter Replacement</p>
                  <p class="text-sm text-muted-foreground">Replaced hydraulic filter • Performed by: John Doe</p>
                </div>
                <div class="text-right">
                  <p class="text-sm font-medium">2024-01-15</p>
                  <p class="text-xs text-muted-foreground">2 weeks ago</p>
                </div>
              </div>

              <div class="flex items-center justify-between p-4 border rounded-lg">
                <div>
                  <p class="font-medium">Oil Change</p>
                  <p class="text-sm text-muted-foreground">Changed hydraulic oil • Performed by: Jane Smith</p>
                </div>
                <div class="text-right">
                  <p class="text-sm font-medium">2024-01-01</p>
                  <p class="text-xs text-muted-foreground">4 weeks ago</p>
                </div>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <UiTabsContent value="diagnostics" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Diagnostic Reports</UiCardTitle>
            <UiCardDescription>Automated diagnostic results and recommendations</UiCardDescription>
          </UiCardHeader>
          <UiCardContent>
            <div class="space-y-4">
              <div class="p-4 border rounded-lg">
                <div class="flex items-center justify-between mb-2">
                  <p class="font-medium">Weekly Health Check</p>
                  <span class="text-sm text-status-success">Passed</span>
                </div>
                <p class="text-sm text-muted-foreground">All systems operating within normal parameters</p>
                <p class="text-xs text-muted-foreground mt-1">Completed: 2024-01-20 14:30</p>
              </div>

              <div class="p-4 border rounded-lg">
                <div class="flex items-center justify-between mb-2">
                  <p class="font-medium">Pressure System Analysis</p>
                  <span class="text-sm text-status-warning">Attention Required</span>
                </div>
                <p class="text-sm text-muted-foreground">Minor pressure fluctuations detected. Recommend monitoring.</p>
                <p class="text-xs text-muted-foreground mt-1">Completed: 2024-01-18 09:15</p>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>

      <UiTabsContent value="docs" class="space-y-6">
        <UiCard>
          <UiCardHeader>
            <UiCardTitle>Documentation</UiCardTitle>
            <UiCardDescription>Manuals, schematics, and technical specifications</UiCardDescription>
          </UiCardHeader>
          <UiCardContent>
            <div class="space-y-4">
              <div class="flex items-center justify-between p-4 border rounded-lg">
                <div class="flex items-center space-x-3">
                  <Icon name="lucide:file-text" class="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p class="font-medium">User Manual</p>
                    <p class="text-sm text-muted-foreground">HYD-001 Operation Guide v2.1</p>
                  </div>
                </div>
                <UiButton variant="outline" size="sm">
                  <Icon name="lucide:download" class="mr-2 h-4 w-4" />
                  Download
                </UiButton>
              </div>

              <div class="flex items-center justify-between p-4 border rounded-lg">
                <div class="flex items-center space-x-3">
                  <Icon name="lucide:image" class="h-5 w-5 text-muted-foreground" />
                  <div>
                    <p class="font-medium">System Schematic</p>
                    <p class="text-sm text-muted-foreground">Wiring and hydraulic diagrams</p>
                  </div>
                </div>
                <UiButton variant="outline" size="sm">
                  <Icon name="lucide:download" class="mr-2 h-4 w-4" />
                  Download
                </UiButton>
              </div>
            </div>
          </UiCardContent>
        </UiCard>
      </UiTabsContent>
    </UiTabs>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const activeTab = ref('overview')

const equipment = computed(() => ({
  id: route.params.id,
  name: 'HYD-001',
  type: 'Pump Station',
  location: 'Building A',
  status: 'online',
  health: 98
}))

const getStatusColor = (status: string) => {
  switch (status) {
    case 'online': return 'bg-status-success'
    case 'warning': return 'bg-status-warning'
    case 'error': return 'bg-status-error'
    case 'offline': return 'bg-muted-foreground'
    default: return 'bg-muted-foreground'
  }
}
</script>
