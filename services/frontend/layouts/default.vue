<template>
  <div class="min-h-screen bg-metal-dark text-white">
    <!-- App Navbar -->
    <UiAppNavbar />

    <!-- Breadcrumb -->
    <UiBreadcrumb />

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <slot />
    </main>

    <!-- Optional Footer -->
    <footer class="border-t border-steel-light/20 mt-auto">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <p class="text-center text-sm text-gray-400">
          {{ t('footer.copyright', { year: new Date().getFullYear() }) }}
        </p>
      </div>
    </footer>
  </div>
</template>

<script setup lang="ts">
const { t } = useI18n()
</script>
