<template>
  <div
    :class="cn(
      'relative w-full rounded-lg border px-4 py-3 text-sm grid items-start gap-y-0.5',
      hasIcon ? 'grid-cols-[calc(var(--spacing)*4)_1fr] gap-x-3' : 'grid-cols-[0_1fr]',
      variant === 'destructive' ? 'text-destructive bg-card' : 'bg-card text-card-foreground',
      className
    )"
    role="alert"
    v-bind="$attrs"
  >
    <slot name="icon" />
    <div class="col-start-2">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { cn } from "./utils";
import { computed, useSlots } from "vue";

interface Props {
  variant?: 'default' | 'destructive';
  className?: string;
}

withDefaults(defineProps<Props>(), {
  variant: 'default',
});

const slots = useSlots();
const hasIcon = computed(() => !!slots.icon);
</script>
