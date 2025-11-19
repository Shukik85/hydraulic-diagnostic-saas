<script setup lang="ts">
import { computed } from 'vue'

const route = useRoute();
const systemId = route.params.systemId;

definePageMeta({ layout: 'dashboard' });
const activeTab = computed(() => {
  if (route.path.startsWith(`/systems/${systemId}/equipments`)) return 'equipments';
  if (route.path.startsWith(`/systems/${systemId}/sensors`)) return 'sensors';
  return '';
});
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">Система #{{ systemId }}</h1>
      <p class="text-gray-600 dark:text-gray-400">Детальная информация о гидравлической системе</p>
    </div>

    <!-- Pill Tabs (Switchers) -->
    <div class="mb-8 flex space-x-2 overflow-x-auto">
      <NuxtLink :to="`/systems/${systemId}/equipments`"
        class="px-5 py-2 rounded-full font-medium whitespace-nowrap transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:z-10 border"
        :class="[
          activeTab === 'equipments'
            ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-500 font-bold'
            : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-gray-700 hover:text-blue-700 hover:border-blue-300',
        ]">
        Оборудование
      </NuxtLink>
      <NuxtLink :to="`/systems/${systemId}/sensors`"
        class="px-5 py-2 rounded-full font-medium whitespace-nowrap transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:z-10 border"
        :class="[
          activeTab === 'sensors'
            ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 border-blue-500 font-bold'
            : 'bg-white dark:bg-gray-800 text-gray-600 dark:text-gray-400 border-gray-200 dark:border-gray-700 hover:text-blue-700 hover:border-blue-300',
        ]">
        Датчики
      </NuxtLink>
    </div>

    <!-- Основное содержимое: статистика и preview -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Статус</p>
            <p class="text-2xl font-bold text-green-600">Активна</p>
          </div>
          <div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:check-circle" class="w-6 h-6 text-green-600" />
          </div>
        </div>
      </div>

      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Давление</p>
            <p class="text-2xl font-bold text-blue-600">150 bar</p>
          </div>
          <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:arrow-trending-up" class="w-6 h-6 text-blue-600" />
          </div>
        </div>
      </div>

      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Температура</p>
            <p class="text-2xl font-bold text-orange-600">45°C</p>
          </div>
          <div class="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:fire" class="w-6 h-6 text-orange-600" />
          </div>
        </div>
      </div>

      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Эффективность</p>
            <p class="text-2xl font-bold text-purple-600">94%</p>
          </div>
          <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:chart-bar" class="w-6 h-6 text-purple-600" />
          </div>
        </div>
      </div>
    </div>

    <!-- Основные действия -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Оборудование -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Оборудование</h3>
            <NuxtLink :to="`/systems/${systemId}/equipments`"
              class="text-blue-600 hover:text-blue-700 text-sm font-medium">Посмотреть всё</NuxtLink>
          </div>
        </div>
        <div class="p-6">
          <div class="text-center py-8">
            <Icon name="heroicons:cog-6-tooth" class="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p class="text-gray-500 dark:text-gray-400 mb-4">Список оборудования системы</p>
            <NuxtLink :to="`/systems/${systemId}/equipments`"
              class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">Управление
              оборудованием</NuxtLink>
          </div>
        </div>
      </div>

      <!-- Датчики -->
      <div class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div class="flex items-center justify-between">
            <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Датчики</h3>
            <NuxtLink :to="`/systems/${systemId}/sensors`"
              class="text-blue-600 hover:text-blue-700 text-sm font-medium">Посмотреть всё</NuxtLink>
          </div>
        </div>
        <div class="p-6">
          <div class="text-center py-8">
            <Icon name="heroicons:cpu-chip" class="w-12 h-12 text-gray-400 mx-auto mb-3" />
            <p class="text-gray-500 dark:text-gray-400 mb-4">Список датчиков системы</p>
            <NuxtLink :to="`/systems/${systemId}/sensors`"
              class="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">Управление
              датчиками</NuxtLink>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
