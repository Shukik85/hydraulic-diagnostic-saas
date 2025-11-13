<template>
  <component
    :is="'label'"
    data-slot="label"
    :class="
      cn(
        'flex items-center gap-2 text-sm leading-none font-semibold select-none',
        'text-text-secondary uppercase letter-spacing-wide',
        'group-data-[disabled=true]:pointer-events-none group-data-[disabled=true]:opacity-50',
        'peer-disabled:cursor-not-allowed peer-disabled:opacity-50',
        className
      )
    "
    v-bind="$attrs"
  >
    <slot />
  </component>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  className?: string;
}

withDefaults(defineProps<Props>(), {});
</script>