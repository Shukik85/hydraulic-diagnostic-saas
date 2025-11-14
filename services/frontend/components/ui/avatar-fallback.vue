<template>
  <div
    :class="[
      'flex size-full items-center justify-center rounded-full metallic-avatar-fallback',
      className
    ]"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>
<script setup lang="ts">// ...Остальной код без изменений...</script>
<style scoped>.metallic-avatar-fallback { background: linear-gradient(135deg, #232b36 70%, #171c22 100%); color: #bbc6d6; font-weight: bold; border: 1.5px solid #4c596f; }</style>