<template>
  <div
    :class="cn('relative h-2 w-full overflow-hidden rounded-full bg-steel-dark border border-steel-medium', className)"
    v-bind="$attrs"
  >
    <div
      :class="cn('h-full w-full flex-1 transition-all duration-300 bg-gradient-to-r from-primary-600 to-primary-500')"
      :style="{ 
        transform: `translateX(-${100 - (value || 0)}%)`,
        boxShadow: '0 0 10px rgba(79, 70, 229, 0.5)'
      }"
    />
  </div>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  value?: number;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  value: 0,
});
</script>