<template>
  <div
    :class="
      cn(
        'text-sm text-text-secondary',
        '[&_p]:leading-relaxed',
        props.className
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

const props = withDefaults(defineProps<Props>(), {})
</script>