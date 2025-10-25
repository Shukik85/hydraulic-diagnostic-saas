<script setup lang="ts">
definePageMeta({ layout: 'dashboard' })
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Отчёты</h1>
      <p class="text-gray-600 dark:text-gray-400">Создание, просмотр и управление аналитическими отчётами</p>
    </div>

    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white">Список отчётов</h2>
        <button class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
          <Icon name="heroicons:plus" class="w-4 h-4 inline mr-2" />Новый отчёт
        </button>
      </div>
      <div class="p-6">
        <div class="text-center py-12">
          <Icon name="heroicons:document-text" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <p class="text-gray-500 dark:text-gray-400">Пока нет отчётов. Создайте первый, чтобы начать.</p>
        </div>
      </div>
    </div>
  </div>
</template>