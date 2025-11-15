<template>
  <h4
    data-slot="card-title"
    :class="cn('leading-none font-bold text-text-primary', props.className)"
    v-bind="$attrs"
  >
    <slot />
  </h4>
</template>

<script setup lang="ts">
import { cn } from './utils'

interface Props {
  className?: string
}

const props = withDefaults(defineProps<Props>(), {})
</script>