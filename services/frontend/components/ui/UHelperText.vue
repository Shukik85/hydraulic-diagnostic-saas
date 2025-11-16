<template>
  <div 
    :class="[
      'flex items-start gap-2 mt-1.5 text-sm',
      variantClass
    ]"
  >
    <!-- Icon -->
    <Icon 
      v-if="showIcon"
      :name="iconName" 
      class="w-4 h-4 mt-0.5 flex-shrink-0"
    />

    <!-- Text Content -->
    <span class="leading-relaxed">
      <slot>{{ text }}</slot>
    </span>
  </div>
</template>

<script setup lang="ts">
interface Props {
  text?: string
  variant?: 'default' | 'success' | 'warning' | 'error'
  icon?: string
  showIcon?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  text: undefined,
  variant: 'default',
  icon: undefined,
  showIcon: true,
})

const iconName = computed(() => {
  if (props.icon) return props.icon

  const icons = {
    default: 'heroicons:light-bulb',
    success: 'heroicons:check-circle',
    warning: 'heroicons:exclamation-triangle',
    error: 'heroicons:x-circle',
  }
  return icons[props.variant]
})

const variantClass = computed(() => {
  const classes = {
    default: 'text-steel-shine',
    success: 'text-success-400',
    warning: 'text-yellow-400',
    error: 'text-red-400',
  }
  return classes[props.variant]
})
</script>
