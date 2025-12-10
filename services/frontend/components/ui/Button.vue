<script setup lang="ts">
import { computed } from 'vue';

interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';
  size?: 'sm' | 'md' | 'lg';
  type?: 'button' | 'submit' | 'reset';
  disabled?: boolean;
  loading?: boolean;
  fullWidth?: boolean;
  icon?: string;
  iconPosition?: 'left' | 'right';
}

const props = withDefaults(defineProps<ButtonProps>(), {
  variant: 'primary',
  size: 'md',
  type: 'button',
  disabled: false,
  loading: false,
  fullWidth: false,
  iconPosition: 'left',
});

const emit = defineEmits<{
  click: [event: MouseEvent];
}>();

const isDisabled = computed(() => props.disabled || props.loading);

const handleClick = (event: MouseEvent) => {
  if (!isDisabled.value) {
    emit('click', event);
  }
};

const buttonClasses = computed(() => {
  const classes = [
    // Base styles
    'inline-flex items-center justify-center gap-2',
    'rounded-md font-medium transition-colors',
    'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
    'disabled:pointer-events-none disabled:opacity-50',
  ];

  // Size variants
  const sizeClasses = {
    sm: 'h-9 px-3 text-sm',
    md: 'h-10 px-4 py-2 text-sm',
    lg: 'h-11 px-8 text-base',
  };
  classes.push(sizeClasses[props.size]);

  // Color variants
  const variantClasses = {
    primary: 'bg-primary text-primary-foreground hover:bg-primary/90',
    secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
    outline:
      'border border-input bg-background hover:bg-accent hover:text-accent-foreground',
    ghost: 'hover:bg-accent hover:text-accent-foreground',
    destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
  };
  classes.push(variantClasses[props.variant]);

  // Full width
  if (props.fullWidth) {
    classes.push('w-full');
  }

  return classes.join(' ');
});
</script>

<template>
  <button
    :type="type"
    :disabled="isDisabled"
    :class="buttonClasses"
    @click="handleClick"
  >
    <!-- Loading spinner -->
    <svg
      v-if="loading"
      class="animate-spin h-4 w-4"
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle
        class="opacity-25"
        cx="12"
        cy="12"
        r="10"
        stroke="currentColor"
        stroke-width="4"
      />
      <path
        class="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>

    <!-- Left icon -->
    <Icon v-if="icon && iconPosition === 'left' && !loading" :name="icon" class="h-4 w-4" />

    <!-- Slot content -->
    <slot />

    <!-- Right icon -->
    <Icon v-if="icon && iconPosition === 'right' && !loading" :name="icon" class="h-4 w-4" />
  </button>
</template>
