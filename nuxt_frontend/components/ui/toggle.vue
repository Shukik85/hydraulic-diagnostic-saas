<template>
  <button
    :class="cn(
      'inline-flex items-center justify-center gap-2 rounded-md text-sm font-medium hover:bg-muted hover:text-muted-foreground disabled:pointer-events-none disabled:opacity-50 data-[state=on]:bg-accent data-[state=on]:text-accent-foreground [&_svg]:pointer-events-none [&_svg:not([class*=\'size-\'])]:size-4 [&_svg]:shrink-0 focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] outline-none transition-[color,box-shadow] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive whitespace-nowrap',
      variant === 'outline' ? 'border border-input bg-transparent hover:bg-accent hover:text-accent-foreground' : 'bg-transparent',
      size === 'sm' ? 'h-8 px-1.5 min-w-8' : size === 'lg' ? 'h-10 px-2.5 min-w-10' : 'h-9 px-2 min-w-9',
      className
    )"
    :data-state="pressed ? 'on' : 'off'"
    @click="toggle"
    v-bind="$attrs"
  >
    <slot />
  </button>
</template>

<script setup lang="ts">
import { cn } from "./utils";
import { computed } from "vue";

interface Props {
  variant?: 'default' | 'outline';
  size?: 'default' | 'sm' | 'lg';
  className?: string;
  modelValue?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'default',
});

const emit = defineEmits<{
  'update:modelValue': [value: boolean];
}>();

const pressed = computed({
  get: () => props.modelValue,
  set: (value: boolean) => emit('update:modelValue', value),
});

const toggle = () => {
  pressed.value = !pressed.value;
};
</script>
