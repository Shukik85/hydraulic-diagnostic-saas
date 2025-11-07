<template>
  <div class="grid gap-6 md:grid-cols-2 lg:grid-cols-2">
    <!-- System Performance Chart -->
    <div class="u-card p-6">
      <div class="flex items-center justify-between mb-4">
        <h3 class="u-h5">System Performance</h3>
        <span class="u-badge u-badge-success text-xs">
          <Icon name="heroicons:check-circle" class="w-3 h-3" />
          Online
        </span>
      </div>
      <ChartLine
        :data="performanceData"
        :show-area="true"
        color="#10b981"
        :show-grid="true"
        class="h-48 w-full"
      />
    </div>

    <!-- Temperature & Pressure -->
    <div class="u-card p-6">
      <h3 class="u-h5 mb-4">Temperature & Pressure Trends</h3>
      <div class="space-y-4">
        <!-- Temperature Sparkline -->
        <div class="flex items-center justify-between">
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-900">Temperature</p>
            <p class="text-xs text-gray-500">Current: {{ currentTemp }}°C</p>
          </div>
          <div class="w-20 h-8">
            <Sparklines :data="temp" color="#f59e0b" />
          </div>
        </div>
        
        <!-- Pressure Sparkline -->
        <div class="flex items-center justify-between">
          <div class="flex-1">
            <p class="text-sm font-medium text-gray-900">Pressure</p>
            <p class="text-xs text-gray-500">Current: {{ currentPressure }} bar</p>
          </div>
          <div class="w-20 h-8">
            <Sparklines :data="pressure" color="#8b5cf6" />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

// Fetch API data correctly
const response = await $fetch('/api/demo/hydraulic-metrics')

// Extract sparkline data from metrics structure
const temp = computed(() => [...(response?.metrics?.temperature?.sparkline || [])])
const pressure = computed(() => [...(response?.metrics?.pressure?.sparkline || [])])
const flow = computed(() => [...(response?.metrics?.flow_rate?.sparkline || [])])
const vibration = computed(() => [...(response?.metrics?.vibration?.sparkline || [])])

// Current values
const currentTemp = computed(() => response?.metrics?.temperature?.current || 0)
const currentPressure = computed(() => response?.metrics?.pressure?.current || 0)

// Performance chart data (derived from metrics)
const performanceData = computed(() => {
  const length = Math.max(temp.value.length, pressure.value.length)
  return Array.from({ length }, (_, i) => ({
    name: `${i * 2}h`,
    value: temp.value[i] || currentTemp.value
  }))
})
</script>