<script setup lang="ts">
const route = useRoute()
const sensorId = route.params.sensorId

definePageMeta({ layout: 'dashboard' })
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Датчик #{{ sensorId }}</h1>
      <p class="text-gray-600 dark:text-gray-400">Паспорт и статус</p>
    </div>

    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      <div class="lg:col-span-2 bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Характеристики</h3>
        <div class="text-gray-600 dark:text-gray-300">Информация о датчике.</div>
      </div>
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 space-y-3">
        <NuxtLink :to="`/sensors/${sensorId}/data`" class="block w-full px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-center">Данные</NuxtLink>
        <NuxtLink :to="`/sensors/${sensorId}/calibration`" class="block w-full px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-800 text-center">Калибровка</NuxtLink>
        <NuxtLink :to="`/sensors/${sensorId}/alerts`" class="block w-full px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 text-center">События</NuxtLink>
      </div>
    </div>
  </div>
</template>