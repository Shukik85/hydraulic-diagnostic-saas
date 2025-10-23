<template>
  <div
    :class="cn(
      'bg-border shrink-0',
      orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
      className
    )"
    v-bind="$attrs"
  />
</template>

<script setup lang="ts">
import { cn } from "./utils";

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
