<template>
  <TabsList
    :class="cn('inline-flex h-10 items-center justify-center', 'rounded-lg p-1 gap-1', 'bg-steel-darker border border-steel-medium', 'shadow-sm', props.className)"
    v-bind="$attrs"
  >
    <slot />
  </TabsList>
</template>
<script setup lang="ts">
import { TabsList } from 'radix-vue'
import { cn } from './utils'
interface Props { className?: string }
const props = withDefaults(defineProps<Props>(), {})
</script>