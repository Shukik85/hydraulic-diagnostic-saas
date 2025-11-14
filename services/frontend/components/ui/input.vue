<template>
  <input
    :type="type"
    data-slot="input"
    :class="
      cn(
        'input-metal',
        'file:text-text-primary placeholder:text-text-muted selection:bg-primary-500 selection:text-white',
        'file:inline-flex file:h-7 file:border-0 file:bg-transparent file:text-sm file:font-medium',
        'disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50',
        className
      )
    "
    v-bind="$attrs"
  />
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  type?: string;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  type: 'text',
});
</script>