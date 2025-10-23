<template>
  <div
    :class="cn(
      'inline-flex h-9 w-fit items-center justify-center rounded-xl p-[3px] flex',
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
