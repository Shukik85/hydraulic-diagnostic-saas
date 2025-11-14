<template>
  <DropdownMenuLabel
    :class="
      cn(
        'px-3 py-1.5 text-xs font-semibold',
        'text-text-secondary uppercase tracking-wider',
        className
      )
    "
    v-bind="$attrs"
  >
    <slot />
  </DropdownMenuLabel>
</template>

<script setup lang="ts">
import { DropdownMenuLabel } from 'radix-vue'
import { cn } from './utils'

interface Props {
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>