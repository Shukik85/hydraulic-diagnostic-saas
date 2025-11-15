<template>
  <button
    :class="[
      'flex w-full cursor-pointer items-center gap-2 rounded-md py-2 px-3 text-sm outline-none',
      'text-text-primary hover:bg-primary-600/10 hover:text-primary-300',
      'focus:bg-primary-600/10 focus:text-primary-300',
      'transition-colors duration-200',
      props.className
    ]"
    @click="emit('select', props.value)"
    v-bind="$attrs"
  >
    <Icon v-if="props.selected" name="lucide:check" class="h-4 w-4 text-primary-500" />
    <slot />
  </button>
</template>

<script setup lang="ts">
import { cn } from './utils'

interface Props {
  value: string
  selected?: boolean
  className?: string
}

const props = withDefaults(defineProps<Props>(), {
  selected: false
})

const emit = defineEmits<{
  select: [value: string]
}>()
</script>