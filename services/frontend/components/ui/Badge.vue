<script setup lang="ts">
import { computed } from 'vue';

interface BadgeProps {
  variant?: 'default' | 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
  size?: 'sm' | 'md' | 'lg';
  dot?: boolean;
  icon?: string;
  outline?: boolean;
  pill?: boolean;
}

const props = withDefaults(defineProps<BadgeProps>(), {
  variant: 'default',
  size: 'md',
  dot: false,
  outline: false,
  pill: false,
});

const badgeClasses = computed(() => {
  const classes = [
    // Base styles
    'inline-flex items-center gap-1.5',
    'font-medium',
    'transition-colors duration-200',
  ];

  // Size variants
  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1.5',
  };
  classes.push(sizeClasses[props.size]);

  // Rounded
  if (props.pill) {
    classes.push('rounded-full');
  } else {
    classes.push('rounded-md');
  }

  // Color variants
  if (props.outline) {
    const outlineVariants = {
      default: 'border-2 border-gray-300 text-gray-700 dark:border-gray-600 dark:text-gray-300',
      primary: 'border-2 border-primary-500 text-primary-700 dark:text-primary-400',
      secondary: 'border-2 border-gray-400 text-gray-700 dark:border-gray-500 dark:text-gray-300',
      success: 'border-2 border-green-500 text-green-700 dark:text-green-400',
      warning: 'border-2 border-amber-500 text-amber-700 dark:text-amber-400',
      error: 'border-2 border-red-500 text-red-700 dark:text-red-400',
      info: 'border-2 border-blue-500 text-blue-700 dark:text-blue-400',
    };
    classes.push(outlineVariants[props.variant]);
  } else {
    const solidVariants = {
      default: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-200',
      primary: 'bg-primary-100 text-primary-800 dark:bg-primary-900/30 dark:text-primary-300',
      secondary: 'bg-gray-200 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
      success: 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300',
      warning: 'bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300',
      error: 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300',
      info: 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300',
    };
    classes.push(solidVariants[props.variant]);
  }

  return classes.join(' ');
});

const dotClasses = computed(() => {
  const dotVariants = {
    default: 'bg-gray-500',
    primary: 'bg-primary-500',
    secondary: 'bg-gray-500',
    success: 'bg-green-500',
    warning: 'bg-amber-500',
    error: 'bg-red-500',
    info: 'bg-blue-500',
  };

  const sizeClass = props.size === 'sm' ? 'h-1.5 w-1.5' : props.size === 'lg' ? 'h-2.5 w-2.5' : 'h-2 w-2';

  return `${sizeClass} ${dotVariants[props.variant]} rounded-full`;
});

const iconSize = computed(() => {
  const sizes = {
    sm: 'h-3 w-3',
    md: 'h-4 w-4',
    lg: 'h-5 w-5',
  };
  return sizes[props.size];
});
</script>

<template>
  <span
    :class="badgeClasses"
    role="status"
    :aria-label="`${variant} badge`"
  >
    <!-- Dot indicator -->
    <span v-if="dot" :class="dotClasses" aria-hidden="true" />

    <!-- Icon -->
    <Icon v-if="icon" :name="icon" :class="iconSize" aria-hidden="true" />

    <!-- Content -->
    <slot />
  </span>
</template>
