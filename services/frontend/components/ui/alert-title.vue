<template>
  <h5
    :class="
      cn(
        'mb-1 font-semibold text-sm',
        'text-text-primary',
        'leading-none tracking-tight',
        className
      )
    "
    v-bind="$attrs"
  >
    <slot />
  </h5>
</template>

<script setup lang="ts">
import { cn } from './utils'

interface Props {
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>