<script setup lang="ts">
const route = useRoute();
const systemId = route.params.systemId;
const equipmentId = route.params.equipmentId;

definePageMeta({
  layout: 'dashboard',
});
</script>

<template>
  <div class="container mx-auto px-4 py-6">
    <div class="mb-8">
      <h1 class="text-2xl font-bold text-gray-900 dark:text-white mb-2">
        Оборудование #{{ equipmentId }}
      </h1>
      <p class="text-gray-600 dark:text-gray-400">
        Детальная информация об оборудовании системы #{{ systemId }}
      </p>
    </div>

    <!-- Статус оборудования -->
    <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
      <div
        class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Состояние</p>
            <p class="text-2xl font-bold text-green-600">Работает</p>
          </div>
          <div class="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:check-circle" class="w-6 h-6 text-green-600" />
          </div>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Нагрузка</p>
            <p class="text-2xl font-bold text-blue-600">78%</p>
          </div>
          <div class="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:chart-pie" class="w-6 h-6 text-blue-600" />
          </div>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Вибрация</p>
            <p class="text-2xl font-bold text-yellow-600">0.8 mm/s</p>
          </div>
          <div class="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:radio" class="w-6 h-6 text-yellow-600" />
          </div>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <p class="text-sm font-medium text-gray-600 dark:text-gray-400">Время работы</p>
            <p class="text-2xl font-bold text-purple-600">142 ч</p>
          </div>
          <div class="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
            <Icon name="heroicons:clock" class="w-6 h-6 text-purple-600" />
          </div>
        </div>
      </div>
    </div>

    <!-- Основные блоки -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <!-- Технические характеристики -->
      <div
        class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
      >
        <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">
            Технические характеристики
          </h3>
        </div>
        <div class="p-6">
          <dl class="space-y-4">
            <div>
              <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Модель</dt>
              <dd class="text-sm text-gray-900 dark:text-white">HYD-{{ equipmentId }}</dd>
            </div>
            <div>
              <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">Система</dt>
              <dd class="text-sm text-gray-900 dark:text-white">SYS-{{ systemId }}</dd>
            </div>
            <div>
              <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">
                Максимальное давление
              </dt>
              <dd class="text-sm text-gray-900 dark:text-white">200 bar</dd>
            </div>
            <div>
              <dt class="text-sm font-medium text-gray-500 dark:text-gray-400">
                Рабочая температура
              </dt>
              <dd class="text-sm text-gray-900 dark:text-white">-20°C до +80°C</dd>
            </div>
          </dl>
        </div>
      </div>

      <!-- Действия -->
      <div
        class="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700"
      >
        <div class="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <h3 class="text-lg font-semibold text-gray-900 dark:text-white">Действия</h3>
        </div>
        <div class="p-6 space-y-4">
          <button
            class="w-full px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center"
          >
            <Icon name="heroicons:cpu-chip" class="w-5 h-5 mr-2" />
            Запустить диагностику
          </button>

          <button
            class="w-full px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center justify-center"
          >
            <Icon name="heroicons:arrow-path" class="w-5 h-5 mr-2" />
            Обновить данные
          </button>

          <button
            class="w-full px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors flex items-center justify-center"
          >
            <Icon name="heroicons:document-arrow-down" class="w-5 h-5 mr-2" />
            Скачать отчёт
          </button>

          <NuxtLink
            :to="`/systems/${systemId}/equipments`"
            class="w-full px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors flex items-center justify-center"
          >
            <Icon name="heroicons:arrow-left" class="w-5 h-5 mr-2" />
            Назад к списку
          </NuxtLink>
        </div>
      </div>
    </div>
  </div>
</template>
