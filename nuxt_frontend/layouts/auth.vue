<script setup lang="ts">
// Auth layout with navigation to home
const colorMode = useColorMode()
</script>

<template>
  <div class="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50 dark:from-gray-900 dark:via-blue-900 dark:to-indigo-900">
    <!-- Top navigation bar -->
    <div class="absolute top-0 left-0 right-0 z-10">
      <div class="container mx-auto px-4 py-6">
        <div class="flex items-center justify-between">
          <!-- Logo/Home link -->
          <NuxtLink to="/" class="flex items-center space-x-3 hover:opacity-80 transition-opacity group">
            <div class="w-10 h-10 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-shadow">
              <Icon name="heroicons:cpu-chip" class="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 class="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Гидравлика ИИ
              </h1>
              <p class="text-xs text-gray-600 dark:text-gray-300">
                Диагностическая платформа
              </p>
            </div>
          </NuxtLink>
          
          <!-- Theme toggle and home link -->
          <div class="flex items-center space-x-3">
            <NuxtLink
              to="/"
              class="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg shadow-sm hover:shadow-md transition-all border border-gray-200 dark:border-gray-700"
            >
              <Icon name="heroicons:home" class="w-4 h-4" />
              <span>На главную</span>
            </NuxtLink>
            
            <button
              @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'"
              class="flex items-center justify-center p-2 text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-lg shadow-sm hover:shadow-md transition-all border border-gray-200 dark:border-gray-700"
              :title="colorMode.preference === 'dark' ? 'Переключить на светлую тему' : 'Переключить на тёмную тему'"
            >
              <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Auth content -->
    <div class="relative z-0">
      <slot />
    </div>
  </div>
</template>