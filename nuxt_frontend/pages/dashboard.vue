<template>
  <div class="space-y-6">
    <!-- Header -->
    <div class="flex items-center justify-between">
      <div>
        <h1 class="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p class="text-muted-foreground">Real-time hydraulic system monitoring and diagnostics</p>
      </div>
      <div class="flex items-center gap-2">
        <UiButton variant="outline" size="sm">
          <Icon name="lucide:download" class="mr-2 h-4 w-4" />
          Export Data
        </UiButton>
        <UiButton size="sm">
          <Icon name="lucide:refresh-cw" class="mr-2 h-4 w-4" />
          Refresh
        </UiButton>
      </div>
    </div>

    <!-- Stats Grid -->
    <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">Active Systems</UiCardTitle>
          <Icon name="lucide:activity" class="h-4 w-4 text-muted-foreground" />
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold">142</div>
          <p class="text-xs text-muted-foreground">
            <span class="text-status-success">+12%</span> from last month
          </p>
        </UiCardContent>
      </UiCard>

      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">Anomalies Detected</UiCardTitle>
          <Icon name="lucide:alert-triangle" class="h-4 w-4 text-muted-foreground" />
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold">8</div>
          <p class="text-xs text-muted-foreground">
            <span class="text-status-success">-23%</span> from last week
          </p>
        </UiCardContent>
      </UiCard>

      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">System Health</UiCardTitle>
          <Icon name="lucide:check-circle" class="h-4 w-4 text-muted-foreground" />
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold">96.4%</div>
          <p class="text-xs text-muted-foreground">
            <span class="text-status-success">+2.1%</span> from yesterday
          </p>
        </UiCardContent>
      </UiCard>

      <UiCard>
        <UiCardHeader class="flex flex-row items-center justify-between space-y-0 pb-2">
          <UiCardTitle class="text-sm font-medium">Avg Response Time</UiCardTitle>
          <Icon name="lucide:clock" class="h-4 w-4 text-muted-foreground" />
        </UiCardHeader>
        <UiCardContent>
          <div class="text-2xl font-bold">1.2s</div>
          <p class="text-xs text-muted-foreground">
            <span class="text-status-success">-0.3s</span> from last week
          </p>
        </UiCardContent>
      </UiCard>
    </div>

    <!-- Charts Row -->
    <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
      <UiCard class="col-span-4">
        <UiCardHeader>
          <UiCardTitle>System Performance</UiCardTitle>
          <UiCardDescription>Real-time sensor data over the last 24 hours</UiCardDescription>
        </UiCardHeader>
        <UiCardContent class="pl-2">
          <div class="h-[300px] w-full">
            <!-- Chart placeholder - would use recharts or similar -->
            <div class="flex h-full items-center justify-center bg-muted/20 rounded-md">
              <div class="text-center">
                <Icon name="lucide:bar-chart-3" class="mx-auto h-12 w-12 text-muted-foreground mb-2" />
                <p class="text-sm text-muted-foreground">Time-series chart will be rendered here</p>
                <p class="text-xs text-muted-foreground mt-1">Pressure, Temperature, Flow Rate</p>
              </div>
            </div>
          </div>
        </UiCardContent>
      </UiCard>

      <UiCard class="col-span-3">
        <UiCardHeader>
          <UiCardTitle>Recent Sessions</UiCardTitle>
          <UiCardDescription>Latest diagnostic sessions</UiCardDescription>
        </UiCardHeader>
        <UiCardContent>
          <div class="space-y-4">
            <div class="flex items-center space-x-4">
              <div class="h-2 w-2 bg-status-success rounded-full"></div>
              <div class="flex-1 space-y-1">
                <p class="text-sm font-medium">HYD-001 Diagnostic Complete</p>
                <p class="text-xs text-muted-foreground">All parameters normal</p>
              </div>
              <div class="text-xs text-muted-foreground">2 min ago</div>
            </div>

            <div class="flex items-center space-x-4">
              <div class="h-2 w-2 bg-status-warning rounded-full"></div>
              <div class="flex-1 space-y-1">
                <p class="text-sm font-medium">HYD-002 Pressure Anomaly</p>
                <p class="text-xs text-muted-foreground">Spike detected at 15:30</p>
              </div>
              <div class="text-xs text-muted-foreground">15 min ago</div>
            </div>

            <div class="flex items-center space-x-4">
              <div class="h-2 w-2 bg-status-info rounded-full"></div>
              <div class="flex-1 space-y-1">
                <p class="text-sm font-medium">HYD-003 Maintenance Due</p>
                <p class="text-xs text-muted-foreground">Filter replacement scheduled</p>
              </div>
              <div class="text-xs text-muted-foreground">1 hour ago</div>
            </div>
          </div>
        </UiCardContent>
      </UiCard>
    </div>

    <!-- Equipment Status Grid -->
    <UiCard>
      <UiCardHeader>
        <UiCardTitle>Equipment Status</UiCardTitle>
        <UiCardDescription>Current status of all monitored hydraulic systems</UiCardDescription>
      </UiCardHeader>
      <UiCardContent>
        <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          <div class="flex items-center space-x-4 p-4 border rounded-lg">
            <div class="h-3 w-3 bg-status-success rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium">HYD-001</p>
              <p class="text-xs text-muted-foreground">Pump Station A</p>
            </div>
            <div class="text-right">
              <p class="text-sm font-medium">98%</p>
              <p class="text-xs text-muted-foreground">Health</p>
            </div>
          </div>

          <div class="flex items-center space-x-4 p-4 border rounded-lg">
            <div class="h-3 w-3 bg-status-warning rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium">HYD-002</p>
              <p class="text-xs text-muted-foreground">Pump Station B</p>
            </div>
            <div class="text-right">
              <p class="text-sm font-medium">76%</p>
              <p class="text-xs text-muted-foreground">Health</p>
            </div>
          </div>

          <div class="flex items-center space-x-4 p-4 border rounded-lg">
            <div class="h-3 w-3 bg-status-success rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium">HYD-003</p>
              <p class="text-xs text-muted-foreground">Pump Station C</p>
            </div>
            <div class="text-right">
              <p class="text-sm font-medium">95%</p>
              <p class="text-xs text-muted-foreground">Health</p>
            </div>
          </div>
        </div>
      </UiCardContent>
    </UiCard>

    <!-- Quick Actions -->
    <UiCard>
      <UiCardHeader>
        <UiCardTitle>Quick Actions</UiCardTitle>
        <UiCardDescription>Common tasks and shortcuts</UiCardDescription>
      </UiCardHeader>
      <UiCardContent>
        <div class="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <UiButton variant="outline" class="h-20 flex-col space-y-2" as-child>
            <NuxtLink to="/equipment">
              <Icon name="lucide:plus" class="h-6 w-6" />
              <span class="text-sm">Add Equipment</span>
            </NuxtLink>
          </UiButton>

          <UiButton variant="outline" class="h-20 flex-col space-y-2" as-child>
            <NuxtLink to="/diagnostics">
              <Icon name="lucide:stethoscope" class="h-6 w-6" />
              <span class="text-sm">Run Diagnostics</span>
            </NuxtLink>
          </UiButton>

          <UiButton variant="outline" class="h-20 flex-col space-y-2" as-child>
            <NuxtLink to="/reports">
              <Icon name="lucide:file-text" class="h-6 w-6" />
              <span class="text-sm">Generate Report</span>
            </NuxtLink>
          </UiButton>

          <UiButton variant="outline" class="h-20 flex-col space-y-2" as-child>
            <NuxtLink to="/chat">
              <Icon name="lucide:message-square" class="h-6 w-6" />
              <span class="text-sm">AI Assistant</span>
            </NuxtLink>
          </UiButton>
        </div>
      </UiCardContent>
    </UiCard>
  </div>
</template>

<script setup lang="ts">
// Dashboard page with comprehensive monitoring widgets
</script>
