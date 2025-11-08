<template>
  <ClientOnly>
    <DigitalTwinViewer />
  </ClientOnly>
</template>

<script setup lang="ts">
// Отключаем auth ТОЛЬКО для этой страницы
definePageMeta({
  layout: false,
  auth: false  // ← Важно!
})
</script>
