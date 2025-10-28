<script setup>
definePageMeta({
  layout: 'dashboard',
});
</script>

<template>
  <div class="container mx-auto px-4 py-8">
    <div class="mb-8">
      <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">Дашборд</h1>
      <p class="text-gray-600 dark:text-gray-400">
        Добро пожаловать в панель управления гидравлическими системами
      </p>
    </div>

    <!-- Key Metrics Cards -->
    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
      <div
        class="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-2xl font-bold text-blue-600 dark:text-blue-400">127</h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">Активных систем</p>
          </div>
          <div class="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <Icon name="heroicons:server-stack" class="w-6 h-6 text-blue-600 dark:text-blue-400" />
          </div>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-2xl font-bold text-green-600 dark:text-green-400">99.94%</h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">Время работы</p>
          </div>
          <div class="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
            <Icon name="heroicons:clock" class="w-6 h-6 text-green-600 dark:text-green-400" />
          </div>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-2xl font-bold text-purple-600 dark:text-purple-400">23</h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">Предотвращено аварий</p>
          </div>
          <div class="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
            <Icon
              name="heroicons:shield-check"
              class="w-6 h-6 text-purple-600 dark:text-purple-400"
            />
          </div>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <div class="flex items-center justify-between">
          <div>
            <h3 class="text-2xl font-bold text-orange-600 dark:text-orange-400">89.00%</h3>
            <p class="text-sm text-gray-600 dark:text-gray-400">Снижение затрат</p>
          </div>
          <div class="p-3 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
            <Icon
              name="heroicons:currency-dollar"
              class="w-6 h-6 text-orange-600 dark:text-orange-400"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div
        class="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Быстрые действия</h3>
        <div class="space-y-3">
          <NuxtLink
            to="/diagnostics"
            class="flex items-center space-x-3 p-3 rounded-lg bg-blue-50 dark:bg-blue-900/30 hover:bg-blue-100 dark:hover:bg-blue-900/50 transition-colors"
          >
            <Icon name="heroicons:play" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span class="text-sm font-medium text-gray-900 dark:text-white"
              >Запустить диагностику</span
            >
          </NuxtLink>
          <NuxtLink
            to="/reports"
            class="flex items-center space-x-3 p-3 rounded-lg bg-green-50 dark:bg-green-900/30 hover:bg-green-100 dark:hover:bg-green-900/50 transition-colors"
          >
            <Icon
              name="heroicons:document-text"
              class="w-5 h-5 text-green-600 dark:text-green-400"
            />
            <span class="text-sm font-medium text-gray-900 dark:text-white">Создать отчёт</span>
          </NuxtLink>
          <NuxtLink
            to="/systems"
            class="flex items-center space-x-3 p-3 rounded-lg bg-purple-50 dark:bg-purple-900/30 hover:bg-purple-100 dark:hover:bg-purple-900/50 transition-colors"
          >
            <Icon name="heroicons:plus" class="w-5 h-5 text-purple-600 dark:text-purple-400" />
            <span class="text-sm font-medium text-gray-900 dark:text-white">Добавить систему</span>
          </NuxtLink>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Последние события</h3>
        <div class="space-y-3">
          <div class="flex items-center space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
            <div class="w-2 h-2 bg-green-500 rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-white">
                Система #127 работает нормально
              </p>
              <p class="text-xs text-gray-600 dark:text-gray-400">2 минуты назад</p>
            </div>
          </div>
          <div class="flex items-center space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
            <div class="w-2 h-2 bg-yellow-500 rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-white">
                Обнаружена аномалия в системе #89
              </p>
              <p class="text-xs text-gray-600 dark:text-gray-400">15 минут назад</p>
            </div>
          </div>
          <div class="flex items-center space-x-3 p-3 rounded-lg bg-gray-50 dark:bg-gray-700">
            <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
            <div class="flex-1">
              <p class="text-sm font-medium text-gray-900 dark:text-white">
                Завершена диагностика системы #45
              </p>
              <p class="text-xs text-gray-600 dark:text-gray-400">1 час назад</p>
            </div>
          </div>
        </div>
      </div>

      <div
        class="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 p-6"
      >
        <h3 class="text-lg font-semibold text-gray-900 dark:text-white mb-4">Состояние ИИ</h3>
        <div class="space-y-4">
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-600 dark:text-gray-400">Модель диагностики</span>
            <span class="text-sm font-medium text-green-600 dark:text-green-400">Активна</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-600 dark:text-gray-400">Точность прогнозов</span>
            <span class="text-sm font-medium text-blue-600 dark:text-blue-400">97.3%</span>
          </div>
          <div class="flex items-center justify-between">
            <span class="text-sm text-gray-600 dark:text-gray-400">Обучение модели</span>
            <span class="text-sm font-medium text-gray-600 dark:text-gray-400">Завершено</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
