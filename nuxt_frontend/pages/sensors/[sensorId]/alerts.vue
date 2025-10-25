<script setup lang="ts">
const route = useRoute()
const sensorId = route.params.sensorId

definePageMeta({ layout: 'dashboard' })
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">События датчика #{{ sensorId }}</h1>
      <p class="text-gray-600 dark:text-gray-400">Алармы, пороги и уведомления</p>
    </div>

    <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
      <div class="text-gray-600 dark:text-gray-300">Журнал событий появится здесь.</div>
    </div>
  </div>
</template>