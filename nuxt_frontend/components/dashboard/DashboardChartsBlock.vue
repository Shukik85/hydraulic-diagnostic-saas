<script setup lang='ts'>
import DashboardCharts from '~/components/dashboard/DashboardCharts.client.vue'
import Sparklines from '~/components/dashboard/Sparklines.client.vue'
</script>
<template>
  <div class="space-y-6">
    <Sparklines />
    <DashboardCharts />
  </div>
</template>
