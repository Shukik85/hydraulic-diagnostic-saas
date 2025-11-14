<template>
  <DropdownMenuItem
    :class="
      cn(
        'relative flex cursor-pointer select-none items-center',
        'rounded-md px-3 py-2 gap-2',
        'text-sm font-medium text-text-primary',
        'outline-none transition-colors duration-200',
        // Hover state
        'hover:bg-primary-600/10 hover:text-primary-300',
        // Focus state
        'focus:bg-primary-600/10 focus:text-primary-300',
        // Active state
        'data-[highlighted]:bg-primary-600/10 data-[highlighted]:text-primary-300',
        // Disabled state
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className
      )
    "
    v-bind="$attrs"
  >
    <slot />
  </DropdownMenuItem>
</template>

<script setup lang="ts">
import { DropdownMenuItem } from 'radix-vue'
import { cn } from './utils'

interface Props {
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>