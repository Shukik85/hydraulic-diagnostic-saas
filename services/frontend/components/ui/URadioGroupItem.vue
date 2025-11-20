<script setup lang="ts">
import { RadioGroupItem, RadioGroupIndicator } from 'radix-vue'

interface Props {
  value: string | number
  disabled?: boolean
  id?: string
}

const props = withDefaults(defineProps<Props>(), {
  disabled: false,
})

// Конвертируем number в string
const stringValue = String(props.value)
</script>

<template>
  <div class="flex items-center gap-2">
    <RadioGroupItem
      :id="id"
      :value="stringValue"
      :disabled="disabled"
      class="radio-item"
    >
      <RadioGroupIndicator class="radio-indicator" />
    </RadioGroupItem>
    <label v-if="id" :for="id" class="cursor-pointer">
      <slot />
    </label>
    <div v-else>
      <slot />
    </div>
  </div>
</template>

<style scoped>
.radio-item {
  @apply w-5 h-5 rounded-full border-2 border-gray-300 flex items-center justify-center;
}

.radio-indicator {
  @apply w-3 h-3 rounded-full bg-blue-600;
}
</style>
