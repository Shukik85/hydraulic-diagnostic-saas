<template>
  <div
    data-slot="card-footer"
    :class="cn('flex items-center px-6 pb-6 [.border-t]:pt-6 [.border-t]:border-steel-medium', props.className)"
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