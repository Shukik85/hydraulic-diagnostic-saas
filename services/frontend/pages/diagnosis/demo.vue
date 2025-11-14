<template>
  <ErrorBoundary>
    <slot />
  </ErrorBoundary>
</template>

<script setup lang="ts">
import ErrorBoundary from '~/components/Error/ErrorBoundary.vue'
</script>
