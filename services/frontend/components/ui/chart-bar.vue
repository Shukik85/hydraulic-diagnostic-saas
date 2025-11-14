<template>
  <div class="h-full w-full metallic-chart rounded-xl bg-steel-darker shadow-lg border border-steel-medium px-3 py-2">
    <VChart 
      ref="chartRef"
      :option="option" 
      autoresize 
      :class="props.class || 'h-40 w-full'"
    />
  </div>
</template>

<script setup lang="ts">
// ...Остальной код без изменений...
</script>

<style scoped>
.metallic-chart {
  background: linear-gradient(120deg, #2b3340 0%, #232731 81%);
  box-shadow: 0 2px 12px 0 rgba(61,72,102,.18), 0 1.5px 0 rgba(110,132,180,.07);
}
</style>