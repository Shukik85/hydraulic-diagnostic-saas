<!-- pages/system-metadata/wizard.vue -->
<template>
  <div class="wizard-page">
    <WizardLayout />
  </div>
</template>

<script setup lang="ts">
// Nuxt 4 auto-imports components from ~/components
// WizardLayout будет автоматически импортирован

// SEO metadata
definePageMeta({
  title: 'Настройка метаданных гидросистемы',
  layout: 'dashboard', // или ваш кастомный layout
});

useSeoMeta({
  title: 'Настройка метаданных гидросистемы',
  description: 'Интерактивный мастер настройки метаданных для гидравлического оборудования'
});
</script>

<style scoped>
.wizard-page {
  min-height: 100vh;
  background: #f9fafb;
  padding: 2rem 0;
}
</style>
