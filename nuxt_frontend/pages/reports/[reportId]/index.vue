<script setup lang="ts">
const route = useRoute()
const reportId = route.params.reportId

definePageMeta({ layout: 'dashboard' })
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Отчёт #{{ reportId }}</h1>
      <p class="text-gray-600 dark:text-gray-400">Итоги и ключевые метрики</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Сводка</h3>
        <div class="text-gray-600 dark:text-gray-300">Данные отчёта появятся здесь.</div>
      </div>
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Действия</h3>
        <NuxtLink :to="`/reports/${reportId}/details`" class="block w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-center">Детали</NuxtLink>
      </div>
    </div>
  </div>
</template>