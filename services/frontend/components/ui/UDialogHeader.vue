<template>
  <div :class="cn('flex flex-col gap-2 text-center sm:text-left', props.className)" v-bind="$attrs">
    <slot />
  </div>
</template>
<script setup lang="ts">
import { cn } from './utils'
interface Props { className?: string }
const props = withDefaults(defineProps<Props>(), {})
</script>