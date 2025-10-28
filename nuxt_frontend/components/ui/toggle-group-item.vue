<template>
  <button
    :class="
      cn(
        'inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium hover:bg-muted hover:text-muted-foreground disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground [&_svg]:pointer-events-none [&_svg:not([class*=\'size-\'])]:size-4 [&_svg]:shrink-0 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] outline-none transition-[color,box-shadow] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive whitespace-nowrap',
        variant === 'outline'
          ? 'border border-input bg-transparent hover:bg-accent hover:text-accent-foreground'
          : 'bg-transparent',
        size === 'sm'
          ? 'h-8 px-1.5 min-w-8'
          : size === 'lg'
            ? 'h-10 px-2.5 min-w-10'
            : 'h-9 px-2 min-w-9',
        'min-w-0 flex-1 shrink-0 rounded-none shadow-none first:rounded-l-md last:rounded-r-md focus:z-10 focus-visible:z-10 border-l-0 first:border-l',
        className
      )
    "
    :data-state="isActive ? 'on' : 'off'"
    @click="handleClick"
    v-bind="$attrs"
  >
    <slot />
  </button>
</template>

<script setup lang="ts">
import { cn } from './utils';
import { computed, inject } from 'vue';

interface Props {
  value: string;
  className?: string;
}

const props = defineProps<Props>();

const group = inject<{
  variant: string;
  size: string;
  modelValue: string | string[];
  multiple: boolean;
  updateValue: (value: string) => void;
}>('toggleGroup');

const variant = computed(() => group?.variant || 'default');
const size = computed(() => group?.size || 'default');

const isActive = computed(() => {
  if (group?.multiple) {
    return Array.isArray(group.modelValue) && group.modelValue.includes(props.value);
  } else {
    return group?.modelValue === props.value;
  }
});

const handleClick = () => {
  group?.updateValue(props.value);
};
</script>
