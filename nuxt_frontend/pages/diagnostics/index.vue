<script setup lang="ts">
definePageMeta({ layout: 'dashboard' })
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Диагностика</h1>
      <p class="text-gray-600 dark:text-gray-400">Управление сессиями диагностики</p>
    </div>

    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div class="text-center py-12">
        <Icon name="heroicons:beaker" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
        <p class="text-gray-500 dark:text-gray-400">Нет активных сессий.</p>
      </div>
    </div>
  </div>
</template>