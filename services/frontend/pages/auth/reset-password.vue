<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100">
    <div class="w-full max-w-md p-8 bg-white rounded-lg shadow-lg">
      <!-- Header -->
      <div class="text-center mb-8">
        <div
          class="mx-auto w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mb-4">
          <Icon name="heroicons:key" class="w-6 h-6 text-white" />
        </div>
        <h1 class="text-2xl font-bold text-gray-900">Смена пароля</h1>
        <p class="text-gray-600 mt-2">Создайте новый надёжный пароль</p>
      </div>

      <!-- Reset Form -->
      <form @submit.prevent="handleReset" class="space-y-6">
        <!-- New Password -->
        <div>
          <label for="newPassword" class="block text-sm font-medium text-gray-700 mb-2">
            Новый пароль
          </label>
          <div class="relative">
            <input id="newPassword" v-model="form.newPassword" :type="showPassword ? 'text' : 'password'" required
              class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
              placeholder="Создайте новый пароль" :disabled="isLoading" />
            <button type="button" @click="showPassword = !showPassword"
              class="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
              :disabled="isLoading">
              <Icon :name="showPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
            </button>
          </div>

          <!-- Password Strength Indicator -->
          <div v-if="form.newPassword" class="mt-3">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm text-gray-600">Надёжность</span>
              <span class="text-xs px-2 py-1 rounded-full font-medium" :class="passwordStrength.color === 'red'
                  ? 'bg-red-100 text-red-800'
                  : passwordStrength.color === 'yellow'
                    ? 'bg-yellow-100 text-yellow-800'
                    : passwordStrength.color === 'green'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-blue-100 text-blue-800'
                ">
                {{ passwordStrength.label }}
              </span>
            </div>
            <div class="w-full bg-gray-200 rounded-full h-2">
              <div class="h-2 rounded-full transition-all duration-300" :class="passwordStrength.color === 'red'
                  ? 'bg-red-500'
                  : passwordStrength.color === 'yellow'
                    ? 'bg-yellow-500'
                    : passwordStrength.color === 'green'
                      ? 'bg-green-500'
                      : 'bg-blue-500'
                " :style="{ width: passwordStrength.score + '%' }"></div>
            </div>
          </div>
        </div>

        <!-- Confirm New Password -->
        <div class="relative">
          <label for="confirmPassword" class="block text-sm font-medium text-gray-700 mb-2">
            Подтверждение пароля
          </label>
          <input id="confirmPassword" v-model="form.confirmPassword" :type="showConfirmPassword ? 'text' : 'password'"
            required
            class="w-full px-4 py-3 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors"
            :class="{ 'border-red-300': passwordMismatch && form.confirmPassword }" placeholder="Повторите новый пароль"
            :disabled="isLoading" />

          <!-- Password Match Status -->
          <div v-if="form.confirmPassword" class="mt-2">
            <div v-if="passwordMismatch" class="flex items-center gap-2 text-sm text-red-600">
              <Icon name="heroicons:x-circle" class="w-4 h-4" />
              <span>Пароли не совпадают</span>
            </div>
            <div v-else class="flex items-center gap-2 text-sm text-green-600">
              <Icon name="heroicons:check-circle" class="w-4 h-4" />
              <span>Пароли совпадают</span>
            </div>
          </div>
        </div>

        <button type="submit"
          class="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-4 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          :disabled="isLoading || passwordMismatch || passwordStrength.score < 60">
          <div v-if="isLoading" class="flex items-center justify-center gap-2">
            <div class="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
            Смена пароля...
          </div>
          <span v-else>Обновить пароль</span>
        </button>
      </form>

      <!-- Back to Login -->
      <div class="mt-6 text-center">
        <NuxtLink to="/auth/login" class="text-sm font-medium text-blue-600 hover:text-blue-500 transition-colors">
          ← Назад ко входу
        </NuxtLink>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, toRef, ref } from 'vue'

import type { UiPasswordStrength } from '~/types/api'

definePageMeta({
  layout: 'auth',
  middleware: ['guest']
})

const router = useRouter()

const form = ref({
  newPassword: '',
  confirmPassword: ''
})

const showPassword = ref(false)
const showConfirmPassword = ref(false)
const isLoading = ref(false)

// Password strength
const passwordStrength = usePasswordStrength(toRef(form.value, 'newPassword'))

// Password match validation
const passwordMismatch = computed(() => {
  return form.value.newPassword !== form.value.confirmPassword
})

const handleReset = async () => {
  if (passwordMismatch.value) {
    return
  }

  isLoading.value = true

  try {
    // Password reset logic here
    console.log('Password reset for:', form.value.newPassword)

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500))

    // Success - redirect to login
    await router.push('/auth/login')
  } catch (error) {
    console.error('Password reset failed:', error)
  } finally {
    isLoading.value = false
  }
}
</script>
