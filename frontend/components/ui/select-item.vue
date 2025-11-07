<template>
  <button
    :class="
      cn(
        'flex w-full cursor-default items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-sm outline-none hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground',
        className
      )
    "
    @click="$emit('select', value)"
    v-bind="$attrs"
  >
    <Icon v-if="selected" name="lucide:check" class="h-4 w-4" />
    <slot />
  </button>
</template>

<script setup lang="ts">
import { cn } from './utils';

interface Props {
  value: string;
  selected?: boolean;
  className?: string;
}

withDefaults(defineProps<Props>(), {
  selected: false,
});

defineEmits<{
  select: [value: string];
}>();
</script>
