<template>
  <button
    :class="
      cn(
        'flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none select-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground',
        variant === 'destructive' &&
          'text-destructive hover:bg-destructive/10 focus:bg-destructive/10',
        inset && 'pl-8',
        className
      )
    "
    v-bind="$attrs"
    @click="$emit('click')"
  >
    <slot />
  </button>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  variant?: 'default' | 'destructive';
  inset?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  variant: 'default',
  inset: false,
});

defineEmits<{
  click: [];
}>();
</script>
