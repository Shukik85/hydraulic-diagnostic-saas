<script setup lang="ts">
definePageMeta({
  layout: 'dashboard'
})
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">
        Гидравлические системы
      </h1>
      <p class="text-gray-600 dark:text-gray-400">
        Управление и мониторинг промышленных гидравлических систем
      </p>
    </div>

    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
      <!-- Заголовок с действиями -->
      <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
        <div class="flex items-center justify-between">
          <h2 class="text-lg font-semibold text-gray-900 dark:text-white">
            Список систем
          </h2>
          <button class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            <Icon name="heroicons:plus" class="w-4 h-4 mr-2 inline" />
            Добавить систему
          </button>
        </div>
      </div>

      <!-- Контент таблицы -->
      <div class="p-6">
        <div class="text-center py-12">
          <Icon name="heroicons:server-stack" class="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 class="text-lg font-medium text-gray-900 dark:text-white mb-2">
            Пока нет систем
          </h3>
          <p class="text-gray-500 dark:text-gray-400 mb-6">
            Добавьте первую гидравлическую систему для начала мониторинга
          </p>
          <button class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
            Добавить систему
          </button>
        </div>
      </div>
    </div>
  </div>
</template>