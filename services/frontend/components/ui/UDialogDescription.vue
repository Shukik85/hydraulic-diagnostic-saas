<template>
  <p :class="cn('text-text-secondary text-sm', props.className)" v-bind="$attrs">
    <slot />
  </p>
</template>
<script setup lang="ts">
import { cn } from './utils'
interface Props { className?: string }
const props = withDefaults(defineProps<Props>(), {})
</script>