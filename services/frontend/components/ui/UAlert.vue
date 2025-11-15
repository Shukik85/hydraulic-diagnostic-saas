<template>
  <div
    :class="
      cn(
        'relative w-full rounded-lg px-4 py-3',
        'border border-steel-medium',
        'bg-steel-darker',
        'grid items-start gap-3',
        hasIcon ? 'grid-cols-[24px_1fr]' : 'grid-cols-1',
        // Variant styles
        props.variant === 'success' && 'border-success-500/30 bg-success-500/5',
        props.variant === 'warning' && 'border-warning-500/30 bg-warning-500/5',
        props.variant === 'error' && 'border-error-500/30 bg-error-500/5',
        props.variant === 'info' && 'border-primary-500/30 bg-primary-500/5',
        props.className
      )
    "
    role="alert"
    v-bind="$attrs"
  >
    <!-- Icon slot -->
    <div v-if="hasIcon" class="flex items-start pt-0.5">
      <slot name="icon" />
    </div>
    
    <!-- Content -->
    <div :class="hasIcon ? 'col-start-2' : ''">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { cn } from './utils'
import { computed, useSlots } from 'vue'

interface Props {
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info'
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default'
})

const slots = useSlots()
const hasIcon = computed(() => !!slots.icon)
</script>