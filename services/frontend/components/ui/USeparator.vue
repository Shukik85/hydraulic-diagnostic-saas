<template>
  <div
    :class="
      cn(
        'bg-steel-medium shrink-0',
        props.orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        props.className
      )
    "
    :style="{
      background: props.orientation === 'horizontal' 
        ? 'linear-gradient(90deg, transparent, #4b5563, transparent)'
        : 'linear-gradient(180deg, transparent, #4b5563, transparent)'
    }"
    v-bind="$attrs"
  />
</template>
<script setup lang="ts">
import { cn } from './utils'
interface Props {
  orientation?: 'horizontal' | 'vertical'
  decorative?: boolean
  className?: string
}
const props = withDefaults(defineProps<Props>(), { orientation: 'horizontal', decorative: true })
</script>