<template>
  <h4 data-slot="card-title" :class="cn('leading-none', className)" v-bind="$attrs">
    <slot />
  </h4>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
