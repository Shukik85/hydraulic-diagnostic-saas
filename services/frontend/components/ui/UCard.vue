<template>
  <div
    data-slot="card"
    :class="cn('card-metal flex flex-col gap-6', props.className)"
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