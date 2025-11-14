<template>
  <div
    :class="cn('flex flex-col-reverse gap-2 sm:flex-row sm:justify-end', className)"
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