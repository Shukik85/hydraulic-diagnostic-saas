<template>
  <div
    :class="cn('flex flex-col gap-2 text-center sm:text-left', className)"
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
