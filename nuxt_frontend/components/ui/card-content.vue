<template>
  <div
    data-slot="card-content"
    :class="cn('px-6 [&:last-child]:pb-6', className)"
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
