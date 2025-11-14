<template>
  <textarea
    :class="
      cn(
        'input-metal resize-none',
        'placeholder:text-text-muted',
        'disabled:cursor-not-allowed disabled:opacity-50',
        className
      )
    "
    v-bind="$attrs"
  />
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>