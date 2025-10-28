<template>
  <div
    :class="cn('relative h-2 w-full overflow-hidden rounded-full bg-primary/20', className)"
    v-bind="$attrs"
  >
    <div
      :class="cn('h-full w-full flex-1 bg-primary transition-all')"
      :style="{ transform: `translateX(-${100 - (value || 0)}%)` }"
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
