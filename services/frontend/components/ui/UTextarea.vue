<script setup lang="ts">
import { cn } from './utils'

interface Props {
  className?: string
}

const props = defineProps<Props>()
</script>

<template>
  <textarea
    :class="
      cn(
        'input-metal resize-none',
        'placeholder:text-text-muted',
        'disabled:cursor-not-allowed disabled:opacity-50',
        props.className
      )
    "
    v-bind="$attrs"
  />
</template>
