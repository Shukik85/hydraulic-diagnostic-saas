<!-- eslint-disable vue/html-self-closing -->
<!-- eslint-disable vue/html-self-closing -->
<script setup lang="ts">
// Authentication layout for login/register pages
// Clean, minimal layout focused on auth forms

useSeoMeta({
  viewport: 'width=device-width, initial-scale=1'
})

// Theme handling
const colorMode = useColorMode()
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
    <!-- Background pattern -->
    <div class="absolute inset-0 overflow-hidden">
      <div class="absolute -top-40 -right-32 w-80 h-80 bg-linear-to-br from-blue-400/20 to-purple-600/20 rounded-full blur-3xl"></div>
      // eslint-disable-next-line vue/html-self-closing, vue/html-self-closing
      <div class="absolute -bottom-32 -left-40 w-80 h-80 bg-linear-to-tr from-green-400/20 to-blue-600/20 rounded-full blur-3xl"></div>
    </div>

    <!-- Theme toggle -->
    <div class="absolute top-6 right-6 z-10">
      <button
        class="p-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg hover:bg-white/20 transition-all duration-200"
        @click="colorMode.preference = colorMode.value === 'dark' ? 'light' : 'dark'"
      >
        <Icon
          :name="colorMode.value === 'dark' ? 'heroicons:sun' : 'heroicons:moon'"
          class="w-5 h-5 text-gray-700 dark:text-gray-200"
        />
      </button>
    </div>

    <!-- Back to home link -->
    <div class="absolute top-6 left-6 z-10">
      <NuxtLink
        to="/"
        class="flex items-center space-x-2 px-4 py-3 bg-white/10 backdrop-blur-sm border border-white/20 rounded-lg hover:bg-white/20 transition-all duration-200 text-gray-700 dark:text-gray-200 font-medium"
      >
        <Icon name="heroicons:arrow-left" class="w-4 h-4" />
        <span class="hidden sm:block">Главная</span>
      </NuxtLink>
    </div>

    <!-- Main content -->
    <div class="relative z-0">
      <slot />
    </div>

    <!-- Footer -->
    <div class="absolute bottom-0 left-0 right-0 z-0">
      <div class="bg-white/5 backdrop-blur-sm border-t border-white/10">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div class="flex items-center justify-between text-sm">
            <div class="text-gray-600 dark:text-gray-300">
              © 2025 Hydraulic Diagnostic SaaS. Все права защищены.
            </div>
            <div class="hidden sm:flex items-center space-x-6">
              <a href="#" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors">
                Политика конфиденциальности
              </a>
              <a href="#" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors">
                Условия сервиса
              </a>
              <a href="#" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors">
                Поддержка
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
