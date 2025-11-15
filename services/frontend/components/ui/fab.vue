<template>
  <UiButton
    :class="[
      'fixed z-50 h-14 w-14 rounded-full transition-all duration-300 hover:scale-110 active:scale-95',
      'bg-gradient-to-br from-primary-600 to-primary-700',
      'shadow-lg shadow-primary-500/40 hover:shadow-xl hover:shadow-primary-500/50',
      'border-2 border-primary-400/20',
      positionClasses,
    ]"
    :size="'icon'"
    v-bind="$attrs"
    @click="$emit('click', $event)"
  >
    <Icon :name="icon" class="h-6 w-6 text-white" />
    <span class="sr-only">{{ label }}</span>
  </UiButton>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  icon?: string
  label?: string
  position?: 'bottom-right' | 'bottom-left' | 'top-right' | 'top-left'
  class?: string
}

const props = withDefaults(defineProps<Props>(), {
  icon: 'lucide:plus',
  label: 'Добавить',
  position: 'bottom-right',
  class: '',
})

const positionClasses = computed(() => {
  const positions = {
    'bottom-right': 'bottom-6 right-6',
    'bottom-left': 'bottom-6 left-6',
    'top-right': 'top-6 right-6',
    'top-left': 'top-6 left-6',
  }
  return positions[props.position]
})

defineEmits<{
  click: [event: Event]
}>()
</script>