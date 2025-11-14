<template>
  <h2 :class="cn('text-lg leading-none font-bold text-primary-400 uppercase tracking-wide', className)" v-bind="$attrs">
    <slot />
  </h2>
</template>

<script setup lang="ts">
import { cn } from './utils'

interface Props {
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>