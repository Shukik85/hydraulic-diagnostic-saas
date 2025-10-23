<template>
  <div
    :class="cn('flex-1 outline-none', className)"
    v-show="active"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
import { cn } from "./utils";

interface Props {
  active?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  active: false,
});
</script>
