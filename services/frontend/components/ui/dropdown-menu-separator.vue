<template>
  <DropdownMenuSeparator
    :class="
      cn(
        '-mx-1 my-1 h-px',
        'bg-gradient-to-r from-transparent via-steel-medium to-transparent',
        className
      )
    "
    v-bind="$attrs"
  />
</template>

<script setup lang="ts">
import { DropdownMenuSeparator } from 'radix-vue'
import { cn } from './utils'

interface Props {
  className?: string
}

withDefaults(defineProps<Props>(), {})
</script>