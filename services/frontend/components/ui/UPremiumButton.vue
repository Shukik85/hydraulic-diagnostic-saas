<script setup lang="ts">
interface Props {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg' | 'xl'
  loading?: boolean
  disabled?: boolean
  icon?: string
  iconPosition?: 'left' | 'right'
  fullWidth?: boolean
  gradient?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  loading: false,
  disabled: false,
  iconPosition: 'left',
  fullWidth: false,
  gradient: true,
})

defineEmits<{ click: [event: MouseEvent] }>()

type VariantKey = NonNullable<Props['variant']>
type SizeKey = NonNullable<Props['size']>

const getVariantClasses = (variant: VariantKey, gradient: boolean): string => {
  if (gradient && variant === 'primary') {
    return 'text-white bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 focus:ring-primary-500 shadow-lg shadow-primary-500/30 hover:shadow-xl hover:shadow-primary-500/40'
  }
  const variantMap: Record<VariantKey, string> = {
    primary: 'text-white bg-primary-600 hover:bg-primary-700 focus:ring-primary-500 shadow-md shadow-primary-500/20',
    secondary: 'text-text-primary bg-steel-dark hover:bg-steel-medium border border-steel-medium focus:ring-steel-light',
    ghost: 'text-primary-400 bg-transparent hover:bg-primary-600/10 focus:ring-primary-500',
    danger: 'text-white bg-error-600 hover:bg-error-700 focus:ring-error-500 shadow-md shadow-error-500/20',
  }
  return variantMap[variant]
}

const getSizeClasses = (size: SizeKey): string => {
  const sizeMap: Record<SizeKey, string> = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base',
    xl: 'px-8 py-4 text-lg',
  }
  return sizeMap[size]
}
</script>

<template>
  <button
    :disabled="disabled || loading"
    :class="[
      'inline-flex items-center justify-center font-bold rounded-lg border border-transparent',
      'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-background-primary',
      'transition-all duration-200',
      'hover:scale-105 active:scale-95',
      'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100',
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