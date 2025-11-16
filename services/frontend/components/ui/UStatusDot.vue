<template>
  <div class="flex items-center gap-2">
    <!-- Animated Dot -->
    <div 
      :class="[
        'w-3 h-3 rounded-full',
        statusClass,
        animated ? 'animate-pulse' : ''
      ]" 
    />

    <!-- Optional Label -->
    <span 
      v-if="label"
      :class="[
        'text-sm font-medium',
        labelColorClass
      ]"
    >
      {{ label }}
    </span>
  </div>
</template>

<script setup lang="ts">
interface Props {
  status: 'success' | 'warning' | 'error' | 'info' | 'offline'
  label?: string
  animated?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  label: undefined,
  animated: true,
})

const statusClass = computed(() => {
  const classes = {
    success: 'bg-success-400 shadow-success-400/50 shadow-lg',
    warning: 'bg-yellow-400 shadow-yellow-400/50 shadow-lg',
    error: 'bg-red-400 shadow-red-400/50 shadow-lg',
    info: 'bg-primary-400 shadow-primary-400/50 shadow-lg',
    offline: 'bg-steel-400 shadow-steel-400/50 shadow-lg',
  }
  return classes[props.status]
})

const labelColorClass = computed(() => {
  const classes = {
    success: 'text-success-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
    info: 'text-primary-400',
    offline: 'text-steel-400',
  }
  return classes[props.status]
})
</script>
