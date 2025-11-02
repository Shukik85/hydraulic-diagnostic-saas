<template>
  <h2 :class="cn('text-lg leading-none font-semibold', className)" v-bind="$attrs">
    <slot />
  </h2>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
