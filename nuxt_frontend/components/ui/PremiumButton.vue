<template>
  <component
    :is="tag"
    :class="buttonClasses"
    :disabled="disabled"
    v-bind="$attrs"
  >
    <Icon v-if="icon" :name="icon" class="w-4 h-4" :class="iconClasses" />
    <slot />
  </component>
</template>

<script setup lang="ts">
interface Props {
  variant?: 'primary' | 'secondary' | 'danger' | 'outline' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  icon?: string
  iconPosition?: 'left' | 'right'
  tag?: string
  fullWidth?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  disabled: false,
  loading: false,
  iconPosition: 'left',
  tag: 'button',
  fullWidth: false
})

const buttonClasses = computed(() => {
  const baseClasses = [
    'inline-flex',
    'items-center',
    'justify-center',
    'font-medium',
    'rounded-md',
    'transition-all',
    'duration-200',
    'ease-in-out',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-offset-2',
    'disabled:opacity-50',
    'disabled:cursor-not-allowed'
  ]
  
  if (props.fullWidth) {
    baseClasses.push('w-full')
  }
  
  // Size classes
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm', 
    lg: 'px-6 py-3 text-base'
  }
  
  // Variant classes
  const variantClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500 shadow-sm',
    secondary: 'bg-gray-200 hover:bg-gray-300 text-gray-900 focus:ring-gray-500 dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-100',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500 shadow-sm',
    outline: 'border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 focus:ring-blue-500',
    ghost: 'bg-transparent hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 focus:ring-blue-500'
  }
  
  return [
    ...baseClasses,
    sizeClasses[props.size],
    variantClasses[props.variant]
  ].join(' ')
})

const iconClasses = computed(() => {
  if (props.iconPosition === 'right') {
    return 'ml-2 -mr-1'
  }
  return 'mr-2 -ml-1'
})
</script>