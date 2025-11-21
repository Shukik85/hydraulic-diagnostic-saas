<script setup lang="ts">
import { ref } from '#imports'

const isMenuOpen = ref(false)
</script>

<template>
  <div class="default-layout">
    <slot />
  </div>
</template>
