<template>
  <ClientOnly>
    <component :is="VChart" :option="option" autoresize :class="$props.class || 'h-40 w-full'" />
  </ClientOnly>
</template>

<script setup lang="ts">
import VChart from 'vue-echarts'

defineProps<{
  option: Record<string, any>
  class?: string
}>()
</script>
