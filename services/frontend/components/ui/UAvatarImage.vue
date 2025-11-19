<template>
  <img :class="[
    'aspect-square size-full object-cover metallic-avatar-image',
    props.className
  ]" v-bind="$attrs" />
</template>
<script setup lang="ts">
interface Props {
  className?: string
}
const props = withDefaults(defineProps<Props>(), {})
</script>
<style scoped>.metallic-avatar-image { border-radius: 9999px; box-shadow: 0 2px 6px #2b334040, 0 1.5px 0 #4c596f12; }</style>