<template>
  <div
    :class="cn('group/toggle-group flex w-fit items-center rounded-md shadow-xs', className)"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
import { cn } from './utils';
import { provide } from 'vue';

interface Props {
  variant?: 'default' | 'outline';
  size?: 'default' | 'sm' | 'lg';
  className?: string;
  modelValue?: string | string[];
  multiple?: boolean;
}

const props = withDefaults(defineProps<Props>(), {
  variant: 'default',
  size: 'default',
  multiple: false,
});

const emit = defineEmits<{
  'update:modelValue': [value: string | string[]];
}>();

provide('toggleGroup', {
  variant: props.variant,
  size: props.size,
  modelValue: props.modelValue,
  multiple: props.multiple,
  updateValue: (value: string) => {
    if (props.multiple) {
      const current = Array.isArray(props.modelValue) ? props.modelValue : [];
      const index = current.indexOf(value);
      if (index > -1) {
        current.splice(index, 1);
      } else {
        current.push(value);
      }
      emit('update:modelValue', [...current]);
    } else {
      emit('update:modelValue', value);
    }
  },
});
</script>
