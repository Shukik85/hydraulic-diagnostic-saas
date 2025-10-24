<script setup lang="ts">
// API configuration
const config = useRuntimeConfig()
const apiBase = config.public.apiBase

// Global meta
useHead({
  htmlAttrs: {
    lang: 'ru'
  },
  meta: [
    { name: 'description', content: 'Профессиональная диагностика гидравлических систем' },
    { name: 'keywords', content: 'гидравлика, диагностика, мониторинг, промышленность, SaaS' },
    { property: 'og:type', content: 'website' },
    { property: 'og:site_name', content: 'Hydraulic Diagnostic SaaS' }
  ]
})
</script>

<template>
  <div id="app">
    <!-- Main app wrapper -->
    <NuxtPage />
  </div>
</template>

<style>
/* Global app styles - minimal base */
#app {
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Loading transitions */
.page-enter-active,
.page-leave-active {
  transition: opacity 0.3s;
}

.page-enter-from,
.page-leave-to {
  opacity: 0;
}
</style>