<script setup lang="ts">
const route = useRoute()
const reportId = route.params.reportId

definePageMeta({ layout: 'dashboard' })
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Детали отчёта #{{ reportId }}</h1>
      <p class="text-gray-600 dark:text-gray-400">Разделы, диаграммы и экспорт</p>
    </div>

    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div class="text-gray-600 dark:text-gray-300">Здесь будет детальный состав отчёта.</div>
    </div>
  </div>
</template>