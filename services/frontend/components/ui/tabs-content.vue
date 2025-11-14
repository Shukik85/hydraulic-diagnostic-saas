<template>
  <TabsContent
    :value="value"
    :class="
      cn(
        'mt-2 flex-1 outline-none',
        'animate-in fade-in-50 duration-200',
        className
      )
    "
    v-bind="$attrs"
  >
    <slot />
  </TabsContent>
</template>

<script setup lang="ts">
import { TabsContent } from 'radix-vue'
import { cn } from './utils'

interface Props {
  value: string
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>