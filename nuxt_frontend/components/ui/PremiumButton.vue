<script setup lang="ts">
// Premium button component with type-safe variants and sizes
interface Props {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  loading?: boolean;
  disabled?: boolean;
  icon?: string;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  gradient?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  loading: false,
  disabled: false,
  iconPosition: 'left',
  fullWidth: false,
  gradient: false,
});

defineEmits<{
  click: [event: MouseEvent];
}>();

// Type-safe variant mapping
type VariantKey = NonNullable<Props['variant']>;
type SizeKey = NonNullable<Props['size']>;

const getVariantClasses = (variant: VariantKey, gradient: boolean): string => {
  if (gradient && variant === 'primary') {
    return 'text-white bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 focus:ring-blue-500 shadow-lg hover:shadow-xl';
  }

  const variantMap: Record<VariantKey, string> = {
    primary:
      'text-white bg-blue-600 hover:bg-blue-700 focus:ring-blue-500 shadow-md hover:shadow-lg',
    secondary:
      'text-gray-700 dark:text-gray-200 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 focus:ring-gray-500',
    ghost:
      'text-blue-600 dark:text-blue-400 bg-transparent hover:bg-blue-50 dark:hover:bg-blue-900/20 focus:ring-blue-500',
    danger: 'text-white bg-red-600 hover:bg-red-700 focus:ring-red-500 shadow-md hover:shadow-lg',
  };

  return variantMap[variant];
};

const getSizeClasses = (size: SizeKey): string => {
  const sizeMap: Record<SizeKey, string> = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
    xl: 'px-8 py-4 text-lg',
  };

  return sizeMap[size];
};
</script>

<template>
  <button
    :disabled="disabled || loading"
    :class="[
      'inline-flex items-center justify-center font-bold rounded-lg border border-transparent focus:outline-none focus:ring-2 focus:ring-offset-2 transition-all duration-200 hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100',
      getVariantClasses(variant || 'primary', gradient),
      getSizeClasses(size || 'md'),
      fullWidth ? 'w-full' : '',
    ]"
    @click="$emit('click', $event)"
  >
    <!-- Loading spinner -->
    <Icon v-if="loading" name="heroicons:arrow-path" class="animate-spin w-4 h-4 mr-2" />

    <!-- Left icon -->
    <Icon v-else-if="icon && iconPosition === 'left'" :name="icon" class="w-4 h-4 mr-2" />

    <!-- Content -->
    <slot />

    <!-- Right icon -->
    <Icon v-if="!loading && icon && iconPosition === 'right'" :name="icon" class="w-4 h-4 ml-2" />
  </button>
</template>
