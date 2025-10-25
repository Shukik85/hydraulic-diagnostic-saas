<script setup lang='ts'>
import DashboardCharts from '~/components/dashboard/DashboardCharts.client.vue'
import Sparklines from '~/components/dashboard/Sparklines.client.vue'

// Fetch BTC-derived demo data JSON from server API route (to avoid exposing CSV logic on client)
const { data, error } = await useFetch('/api/demo/hydraulic-metrics')

const temp = computed(() => data.value?.sparklines?.temperature || [45,46,47,46,48,49,47])
const pressure = computed(() => data.value?.sparklines?.pressure || [150,149,151,152,149,150,151])
const flow = computed(() => data.value?.sparklines?.flow_rate || [90,92,91,93,94,92,95])
const vibration = computed(() => data.value?.sparklines?.vibration || [1.1,1.2,1.0,1.3,1.4,1.2,1.1])
</script>
<template>
  <div class="space-y-6">
    <Sparklines :temp="temp" :pressure="pressure" :flow="flow" :vibration="vibration" />
    <DashboardCharts />
  </div>
</template>
