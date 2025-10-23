<template>
  <button
    :class="cn(
      'inline-flex h-[calc(100%-1px)] flex-1 items-center justify-center gap-1.5 rounded-xl border border-transparent px-2 py-1 text-sm font-medium whitespace-nowrap transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50',
      active ? 'bg-card text-foreground' : 'text-muted-foreground hover:text-foreground',
      className
    )"
    @click="$emit('click', value)"
    v-bind="$attrs"
  >
    <slot />
  </button>
</template>

<script setup lang="ts">
import { cn } from "./utils";

interface Props {
  value: string;
  active?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  active: false,
});

defineEmits<{
  click: [value: string];
}>();
</script>
