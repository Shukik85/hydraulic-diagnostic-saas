<script setup lang="ts">
// Minimal auth layout - just centered container with theme/language controls
</script>

<template>
  <div class="relative min-h-screen flex items-center justify-center bg-gradient-to-br from-primary-50 via-white to-primary-100 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800">
    <!-- Floating Controls (top-right) -->
    <div class="absolute right-4 top-4 z-10 flex items-center gap-2">
      <!-- Language Switcher -->
      <LanguageSwitcher />

      <!-- Theme Switcher -->
      <ThemeSwitcher />
    </div>

    <!-- Main Content (centered) -->
    <div class="w-full max-w-md px-4">
      <slot />
    </div>
  </div>
</template>
