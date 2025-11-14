<template>
  <TabsRoot
    v-model="localValue"
    :class="cn('flex flex-col gap-3', className)"
    v-bind="$attrs"
  >
    <slot />
  </TabsRoot>
</template>

<script setup lang="ts">
import { TabsRoot } from 'radix-vue'
import { ref, watch } from 'vue'
import { cn } from './utils'

interface Props {
  modelValue?: string
  className?: string
}

const props = withDefaults(defineProps<Props>(), {})

const emit = defineEmits(['update:modelValue'])

const localValue = ref(props.modelValue)

watch(() => props.modelValue, (newVal) => {
  localValue.value = newVal
})

watch(localValue, (newVal) => {
  emit('update:modelValue', newVal)
})
</script>