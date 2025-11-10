<template>
  <p data-slot="card-description" :class="cn('text-muted-foreground', className)" v-bind="$attrs">
    <slot />
  </p>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>
