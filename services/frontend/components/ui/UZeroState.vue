<template>
  <div class="flex flex-col items-center justify-center py-20 px-6 text-center">
    <!-- Icon Circle -->
    <div class="mb-6 w-24 h-24 rounded-full flex items-center justify-center" :class="iconBgClass">
      <Icon :name="iconName" class="w-12 h-12" :class="iconColorClass" />
    </div>

    <!-- Title -->
    <h3 class="text-2xl font-bold text-white mb-2">
      {{ title }}
    </h3>

    <!-- Description -->
    <p class="text-steel-shine mb-8 max-w-md">
      {{ description }}
    </p>

    <!-- Action Button -->
    <UButton v-if="showAction" size="lg" @click="$emit('action')">
      <Icon :name="actionIcon" class="w-5 h-5 mr-2" />
      {{ actionText }}
    </UButton>

    <!-- Secondary Action (optional) -->
    <button v-if="secondaryText" class="mt-4 text-sm text-primary-400 hover:text-primary-300 transition-colors"
      @click="$emit('secondary-action')">
      {{ secondaryText }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Props {
  iconName: string
  title: string
  description: string
  actionIcon?: string
  actionText?: string
  secondaryText?: string
  variant?: 'primary' | 'success' | 'warning' | 'info'
  showAction?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  actionIcon: 'heroicons:plus',
  actionText: 'Начать',
  variant: 'primary',
  showAction: true,
  secondaryText: undefined,
})

defineEmits<{
  action: []
  'secondary-action': []
}>()

const iconBgClass = computed(() => {
  const variants = {
    primary: 'bg-primary-600/10',
    success: 'bg-success-600/10',
    warning: 'bg-yellow-600/10',
    info: 'bg-blue-600/10',
  }
  return variants[props.variant]
})

const iconColorClass = computed(() => {
  const variants = {
    primary: 'text-primary-400',
    success: 'text-success-400',
    warning: 'text-yellow-400',
    info: 'text-blue-400',
  }
  return variants[props.variant]
})
</script>
