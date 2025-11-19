<template>
  <div
    :class="[
      'relative flex size-10 shrink-0 overflow-hidden rounded-full metallic-avatar shadow shadow-steel-medium/40',
      props.className
    ]"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>
<script setup lang="ts">
interface Props {
  className?: string
}
const props = withDefaults(defineProps<Props>(), {})
</script>
<style scoped>.metallic-avatar { background: linear-gradient(135deg, #232b36 70%, #171c22 100%); border: 2px solid #424c5b; }</style>