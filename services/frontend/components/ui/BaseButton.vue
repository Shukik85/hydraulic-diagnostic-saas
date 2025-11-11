<template>
  <button :type="type" :class="buttonClasses" :disabled="disabled || loading" @click="handleClick">
    <Icon v-if="loading" name="svg-spinners:ring-resize" class="animate-spin" />
    <Icon v-else-if="icon" :name="icon" />
    <slot v-if="!loading"></slot>
  </button>
</template>

<script setup lang="ts">
type Variant = 'primary' | 'secondary' | 'danger' | 'success' | 'ghost'
type Size = 'xs' | 'sm' | 'md' | 'lg'

interface Props {
  variant?: Variant
  size?: Size
  icon?: string
  loading?: boolean
  disabled?: boolean
  type?: 'button' | 'submit' | 'reset'
  fullWidth?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'primary',
  size: 'md',
  type: 'button',
  fullWidth: false
})

const emit = defineEmits<{
  click: [event: MouseEvent]
}>()

const variantClasses: Record<Variant, string> = {
  primary: 'bg-hydraulic-500 hover:bg-hydraulic-600 text-white border-transparent',
  secondary: 'bg-industrial-100 hover:bg-industrial-200 text-industrial-900 border-industrial-300 dark:bg-industrial-700 dark:hover:bg-industrial-600 dark:text-industrial-50',
  danger: 'bg-red-500 hover:bg-red-600 text-white border-transparent',
  success: 'bg-green-500 hover:bg-green-600 text-white border-transparent',
  ghost: 'bg-transparent hover:bg-industrial-100 text-industrial-700 border-transparent dark:hover:bg-industrial-800 dark:text-industrial-300'
}

const sizeClasses: Record<Size, string> = {
  xs: 'px-2 py-1 text-xs gap-1',
  sm: 'px-3 py-1.5 text-sm gap-1.5',
  md: 'px-4 py-2 text-sm gap-2',
  lg: 'px-6 py-3 text-base gap-2'
}

const buttonClasses = computed(() => [
  'inline-flex items-center justify-center',
  'font-medium rounded-button',
  'border transition-all duration-200',
  'focus:outline-none focus:ring-2 focus:ring-hydraulic-500 focus:ring-offset-2',
  'disabled:opacity-50 disabled:cursor-not-allowed',
  variantClasses[props.variant],
  sizeClasses[props.size],
  {
    'w-full': props.fullWidth
  }
])

function handleClick(event: MouseEvent) {
  if (!props.disabled && !props.loading) {
    emit('click', event)
  }
}
</script>