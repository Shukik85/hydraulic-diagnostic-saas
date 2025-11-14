<template>
  <div :class="['flex gap-4', orientation === 'vertical' ? 'flex-col' : 'flex-row', className]">
    <slot />
  </div>
</template>
<script setup lang="ts">
interface Props {
  modelValue?: string | number
  orientation?: 'horizontal' | 'vertical'
  className?: string
}
const props = withDefaults(defineProps<Props>(), {
  orientation: 'horizontal',
})
const emit = defineEmits(['update:modelValue'])
</script>