<template>
  <div
    :class="cn('relative flex size-10 shrink-0 overflow-hidden rounded-full', className)"
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
