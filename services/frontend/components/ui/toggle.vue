<script setup lang="ts">
interface Props {
  variant?: 'default' | 'outline'
  size?: 'sm' | 'md' | 'lg'
  pressed?: boolean
  className?: string
}

interface Emits {
  (e: 'update:pressed', value: boolean): void
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'md',
  pressed: false
})

const emit = defineEmits<Emits>()

const toggle = () => {
  emit('update:pressed', !props.pressed)
}
</script>

<template>
  <button
    :class="[
      'inline-flex items-center justify-center gap-2 rounded-lg text-sm font-bold select-none transition-all duration-200 outline-none',
      props.variant === 'outline'
        ? 'border border-steel-medium bg-transparent hover:bg-steel-dark text-primary-400 hover:text-primary-300'
        : 'bg-steel-dark text-text-primary hover:bg-primary-500/10',
      props.pressed
        ? 'bg-primary-600/90 text-white shadow focus:ring-2 focus:ring-primary-600 focus:ring-offset-2'
        : '',
      props.size === 'sm' ? 'h-8 px-2 min-w-8' : props.size === 'lg' ? 'h-10 px-4 min-w-10' : 'h-9 px-3 min-w-9',
      'disabled:opacity-50 disabled:cursor-not-allowed',
      props.className
    ]"
    :data-state="props.pressed ? 'on' : 'off'"
    @click="toggle"
    v-bind="$attrs"
  >
    <slot />
  </button>
</template>
