<template>
  <div
    :class="cn(
      'col-start-2 line-clamp-1 min-h-4 font-medium tracking-tight',
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
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
