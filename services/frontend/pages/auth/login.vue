<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
    <div class="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
      <!-- Header -->
      <div class="text-center mb-8">
        <div
          class="mx-auto w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mb-4">
          <Icon name="heroicons:cpu-chip" class="w-6 h-6 text-white" />
        </div>
        <h1 class="text-2xl font-bold text-gray-900">Вход в систему</h1>
        <p class="text-gray-600 mt-2">Диагностика гидравлических систем</p>
      </div>

      <!-- Error Alert -->
      <div v-if="authStore.error" class="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
        <div class="flex items-center gap-3">
          <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-red-500 flex-shrink-0" />
          <p class="text-red-800 text-sm">{{ authStore.error }}</p>
        </div>
      </div>

      <!-- Login Form -->
      <form @submit.prevent="handleLogin" class="space-y-6">
        <div>
          <label for="email" class="block text-sm font-medium text-gray-700 mb-2">
            Email адрес
          </label>
          <input id="email" v-model="form.email" type="email" required
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            placeholder="user@company.com" :disabled="authStore.loading" />
        </div>

        <div>
          <label for="password" class="block text-sm font-medium text-gray-700 mb-2">
            Пароль
          </label>
          <div class="relative">
            <input id="password" v-model="form.password" :type="showPassword ? 'text' : 'password'" required
              class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              placeholder="Введите пароль" :disabled="authStore.loading" />
            <button type="button" @click="showPassword = !showPassword"
              class="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
              :disabled="authStore.loading">
              <Icon :name="showPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
            </button>
          </div>
        </div>

        <button type="submit"
          class="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          :disabled="authStore.loading">
          <div v-if="authStore.loading" class="flex items-center justify-center gap-2">
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            Вход...
          </div>
          <span v-else>Войти</span>
        </button>
      </form>

      <!-- Demo credentials -->
      <div class="mt-6 p-4 bg-gray-50 rounded-lg">
        <p class="text-sm font-medium text-gray-700 mb-2">Тестовые данные:</p>
        <div class="text-xs text-gray-600 space-y-1">
          <p><span class="font-mono">admin@hydraulic.ai</span> / <span class="font-mono">admin123</span></p>
          <p><span class="font-mono">engineer@hydraulic.ai</span> / <span class="font-mono">eng123</span></p>
        </div>
      </div>

      <!-- Register Link -->
      <div class="mt-6 text-center">
        <p class="text-sm text-gray-600">
          Нет аккаунта?
          <NuxtLink to="/auth/register" class="font-medium text-blue-600 hover:text-blue-500 transition-colors">
            Зарегистрироваться
          </NuxtLink>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">

// Redirect if already authenticated
definePageMeta({
  layout: 'auth' as const,
  middleware: ['guest']
})

const authStore = useAuthStore()
const router = useRouter()

const form = ref({
  email: '',
  password: ''
})

const showPassword = ref(false)

const handleLogin = async () => {
  try {
    await authStore.login(form.value)
    await router.push('/dashboard')
  } catch (error) {
    console.error('Login failed:', error)
  }
}
</script>
