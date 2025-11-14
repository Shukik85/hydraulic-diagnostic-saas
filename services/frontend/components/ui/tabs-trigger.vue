<template>
  <TabsTrigger
    :value="value"
    :class="
      cn(
        'inline-flex items-center justify-center gap-2',
        'px-4 py-2 rounded-md',
        'text-sm font-medium whitespace-nowrap',
        'transition-all duration-200',
        // Default state
        'text-text-secondary bg-transparent',
        'hover:text-primary-300 hover:bg-steel-dark',
        // Active state
        'data-[state=active]:text-white',
        'data-[state=active]:bg-primary-600',
        'data-[state=active]:shadow-[0_0_12px_rgba(79,70,229,0.4)]',
        // Focus state
        'focus-visible:outline-none focus-visible:ring-2',
        'focus-visible:ring-primary-500/30 focus-visible:ring-offset-2',
        'focus-visible:ring-offset-background-primary',
        // Disabled state
        'disabled:pointer-events-none disabled:opacity-50',
        className
      )
    "
    v-bind="$attrs"
  >
    <slot />
  </TabsTrigger>
</template>

<script setup lang="ts">
import { TabsTrigger } from 'radix-vue'
import { cn } from './utils'

interface Props {
  value: string
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>