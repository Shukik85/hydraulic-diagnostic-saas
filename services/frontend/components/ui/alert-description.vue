<template>
  <div
    :class="
      cn(
        'text-sm text-text-secondary',
        '[&_p]:leading-relaxed',
        className
      )
    "
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
import { cn } from './utils'

interface Props {
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>