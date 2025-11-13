<template>
  <div
    :class="
      cn(
        'bg-steel-medium shrink-0',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className
      )
    "
    :style="{
      background: orientation === 'horizontal' 
        ? 'linear-gradient(90deg, transparent, #4b5563, transparent)'
        : 'linear-gradient(180deg, transparent, #4b5563, transparent)'
    }"
    v-bind="$attrs"
  />
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  orientation?: 'horizontal' | 'vertical';
  decorative?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  orientation: 'horizontal',
  decorative: true,
});
</script>