<template>
  <div class="u-dialog-content">
    <slot />
  </div>
</template>

<script setup lang="ts">
// Dialog content wrapper component for modals
</script>

<style scoped>
.u-dialog-content {
  max-width: 100%;
  position: relative;
  background: var(--color-background, #fff);
  border-radius: var(--radius-lg, 12px);
  padding: var(--space-24, 24px);
}

@media (prefers-color-scheme: dark) {
  .u-dialog-content {
    background: var(--color-surface, #1e293b);
  }
}

[data-color-scheme="dark"] .u-dialog-content {
  background: var(--color-surface, #1e293b);
}
</style>
