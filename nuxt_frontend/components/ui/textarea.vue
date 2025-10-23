<template>
  <textarea
    :class="cn(
      'resize-none border border-input bg-input-background px-3 py-2 text-base rounded-md transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 placeholder:text-muted-foreground',
      className
    )"
    v-bind="$attrs"
  />
</template>

<script setup lang="ts">
import { cn } from "./utils";

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
