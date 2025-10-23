<template>
  <div
    :class="cn(
      'px-2 py-1.5 text-sm font-medium',
      inset && 'pl-8',
      className
    )"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
import { cn } from "./utils";

interface Props {
  inset?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  inset: false,
});
</script>
