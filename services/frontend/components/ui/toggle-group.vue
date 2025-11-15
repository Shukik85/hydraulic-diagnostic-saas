<script setup lang="ts">
interface Props {
  className?: string
}

const props = defineProps<Props>()
</script>

<template>
  <div :class="[
      'group/toggle-group flex w-fit items-center rounded-lg shadow border border-steel-medium bg-steel-darker',
      'p-1 gap-1',
      props.className
    ]"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>
