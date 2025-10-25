<script setup lang="ts">
// Default layout with home navigation
const authStore = useAuthStore()
const colorMode = useColorMode()
</script>

<template>
  <div class="min-h-screen bg-gray-50 dark:bg-gray-900">
    <!-- Simple top bar -->
    <div class="border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-sm">
      <div class="container mx-auto px-4">
        <div class="flex items-center justify-between h-16">
          <!-- Logo/Home -->
          <NuxtLink to="/" class="flex items-center space-x-3 hover:opacity-80 transition-opacity">
            <div class="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center">
              <Icon name="heroicons:cpu-chip" class="w-4 h-4 text-white" />
            </div>
            <div>
              <h1 class="text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                Гидравлика ИИ
              </h1>
            </div>
          </NuxtLink>
          
          <!-- Actions -->
          <div class="flex items-center space-x-3">
            <button
              @click="colorMode.preference = colorMode.preference === 'dark' ? 'light' : 'dark'"
              class="p-2 text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
            >
              <Icon :name="colorMode.preference === 'dark' ? 'heroicons:sun' : 'heroicons:moon'" class="w-5 h-5" />
            </button>
            
            <div v-if="authStore.isAuthenticated" class="flex items-center space-x-3">
              <NuxtLink
                to="/dashboard"
                class="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                Дашборд
              </NuxtLink>
            </div>
            
            <div v-else class="flex items-center space-x-2">
              <NuxtLink
                to="/auth/login"
                class="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
              >
                Войти
              </NuxtLink>
              <NuxtLink
                to="/auth/register"
                class="px-4 py-2 text-sm font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg hover:shadow-xl"
              >
                Регистрация
              </NuxtLink>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Page content -->
    <main>
      <slot />
    </main>
  </div>
</template>