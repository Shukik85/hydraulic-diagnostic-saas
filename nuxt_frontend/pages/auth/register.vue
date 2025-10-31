<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
    <div class="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
      <!-- Header -->
      <div class="text-center mb-8">
        <div class="mx-auto w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mb-4">
          <Icon name="heroicons:user-plus" class="w-6 h-6 text-white" />
        </div>
        <h1 class="text-2xl font-bold text-gray-900">Регистрация</h1>
        <p class="text-gray-600 mt-2">Создайте аккаунт для доступа к платформе</p>
      </div>

      <!-- Registration Form -->
      <form @submit.prevent="handleRegister" class="space-y-6">
        <!-- Name Fields -->
        <div class="grid grid-cols-2 gap-4">
          <div>
            <label for="firstName" class="block text-sm font-medium text-gray-700 mb-2">
              Имя
            </label>
            <input
              id="firstName"
              v-model="form.firstName"
              type="text"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              placeholder="Иван"
              :disabled="isLoading"
            />
          </div>
          <div>
            <label for="lastName" class="block text-sm font-medium text-gray-700 mb-2">
              Фамилия
            </label>
            <input
              id="lastName"
              v-model="form.lastName"
              type="text"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              placeholder="Петров"
              :disabled="isLoading"
            />
          </div>
        </div>

        <!-- Email -->
        <div>
          <label for="email" class="block text-sm font-medium text-gray-700 mb-2">
            Email адрес
          </label>
          <input
            id="email"
            v-model="form.email"
            type="email"
            required
            class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            placeholder="ivan.petrov@company.com"
            :disabled="isLoading"
          />
        </div>

        <!-- Password -->
        <div>
          <label for="password" class="block text-sm font-medium text-gray-700 mb-2">
            Пароль
          </label>
          <div class="relative">
            <input
              id="password"
              v-model="form.password"
              :type="showPassword ? 'text' : 'password'"
              required
              class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              placeholder="Создайте надёжный пароль"
              :disabled="isLoading"
            />
            <button
              type="button"
              @click="showPassword = !showPassword"
              class="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
              :disabled="isLoading"
            >
              <Icon :name="showPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
            </button>
          </div>
          
          <!-- Password Strength Indicator -->
          <div v-if="form.password" class="mt-3">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm text-gray-600">Надёжность пароля</span>
              <span 
                class="text-xs px-2 py-1 rounded-full font-medium"
                :class="
                  passwordStrength.color === 'red'
                    ? 'bg-red-100 text-red-800'
                    : passwordStrength.color === 'yellow'
                    ? 'bg-yellow-100 text-yellow-800'
                    : passwordStrength.color === 'green'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-blue-100 text-blue-800'
                "
              >
                {{ passwordStrength.label }}
              </span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
              <div
                class="h-2 rounded-full transition-all duration-300"
                :class="
                  passwordStrength.color === 'red'
                    ? 'bg-red-500'
                    : passwordStrength.color === 'yellow'
                    ? 'bg-yellow-500'
                    : passwordStrength.color === 'green'
                    ? 'bg-green-500'
                    : 'bg-blue-500'
                "
                :style="{ width: passwordStrength.score + '%' }"
              ></div>
            </div>
          </div>
        </div>

        <!-- Confirm Password -->
        <div>
          <label for="confirmPassword" class="block text-sm font-medium text-gray-700 mb-2">
            Подтвердите пароль
          </label>
          <input
            id="confirmPassword"
            v-model="form.confirmPassword"
            :type="showConfirmPassword ? 'text' : 'password'"
            required
            class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            :class="{ 'border-red-300': passwordMismatch && form.confirmPassword }"
            placeholder="Повторите пароль"
            :disabled="isLoading"
          />
          <button
            type="button"
            @click="showConfirmPassword = !showConfirmPassword"
            class="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
            :disabled="isLoading"
          >
            <Icon :name="showConfirmPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
          </button>
          
          <!-- Password Match Indicator -->
          <div v-if="form.confirmPassword && passwordMismatch" class="mt-2 flex items-center gap-2 text-sm text-red-600">
            <Icon name="heroicons:x-circle" class="w-4 h-4" />
            <span>Пароли не совпадают</span>
          </div>
          <div v-else-if="form.confirmPassword && !passwordMismatch" class="mt-2 flex items-center gap-2 text-sm text-green-600">
            <Icon name="heroicons:check-circle" class="w-4 h-4" />
            <span>Пароли совпадают</span>
          </div>
        </div>

        <!-- Terms Agreement -->
        <div class="flex items-start gap-3">
          <input
            id="agreeTerms"
            v-model="form.agreeTerms"
            type="checkbox"
            required
            class="mt-1 w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
            :disabled="isLoading"
          />
          <label for="agreeTerms" class="text-sm text-gray-600">
            Я соглашаюсь с
            <a href="#" class="text-blue-600 hover:text-blue-500 hover:underline">пользовательским соглашением</a>
            и
            <a href="#" class="text-blue-600 hover:text-blue-500 hover:underline">политикой конфиденциальности</a>
          </label>
        </div>

        <button
          type="submit"
          class="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          :disabled="isLoading || passwordMismatch || !form.agreeTerms"
        >
          <div v-if="isLoading" class="flex items-center justify-center gap-2">
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            Регистрация...
          </div>
          <span v-else>Создать аккаунт</span>
        </button>
      </form>

      <!-- Login Link -->
      <div class="mt-6 text-center">
        <p class="text-sm text-gray-600">
          Уже есть аккаунт?
          <NuxtLink to="/auth/login" class="font-medium text-blue-600 hover:text-blue-500 transition-colors">
            Войти
          </NuxtLink>
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { UiPasswordStrength } from '~/types/api'

// Redirect if already authenticated
definePageMeta({
  layout: 'auth',
  middleware: ['guest']
})

const router = useRouter()

const form = ref({
  firstName: '',
  lastName: '',
  email: '',
  password: '',
  confirmPassword: '',
  agreeTerms: false
})

const showPassword = ref(false)
const showConfirmPassword = ref(false)
const isLoading = ref(false)

// Password strength
const passwordStrength = usePasswordStrength(toRef(form.value, 'password'))

// Password match validation
const passwordMismatch = computed(() => {
  return form.value.password !== form.value.confirmPassword
})

const handleRegister = async () => {
  if (passwordMismatch.value || !form.value.agreeTerms) {
    return
  }
  
  isLoading.value = true
  
  try {
    // Registration logic here
    console.log('Registration data:', form.value)
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // Success - redirect to dashboard
    await router.push('/dashboard')
  } catch (error) {
    console.error('Registration failed:', error)
  } finally {
    isLoading.value = false
  }
}
</script>