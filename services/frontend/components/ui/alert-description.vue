<template>
  <div
    :class="
      cn('text-muted-foreground col-start-2 grid justify-items-start gap-1 text-sm', className)
    "
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
