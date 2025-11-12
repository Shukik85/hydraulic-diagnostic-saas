/**
 * Login Page - User authentication
 * 
 * Features:
 * - Nuxt UI form components
 * - Dark mode support
 * - i18n ready
 * - Loading states
 * - Error handling
 * - Password visibility toggle
 * - Demo credentials display
 */
<script setup lang="ts">
import type { FormSubmitEvent } from '#ui/types'

// Page meta
definePageMeta({
  layout: 'auth',
  middleware: ['guest']
})

// Composables
const authStore = useAuthStore()
const router = useRouter()
const toast = useToast()
const { t } = useI18n()

// Form state
interface LoginForm {
  email: string
  password: string
}

const state = reactive<LoginForm>({
  email: '',
  password: ''
})

const isLoading = ref(false)
const showPassword = ref(false)

/**
 * Handle login form submission
 */
async function onSubmit(event: FormSubmitEvent<LoginForm>) {
  isLoading.value = true

  try {
    await authStore.login({
      email: event.data.email,
      password: event.data.password
    })

    toast.add({
      title: t('auth.login.success'),
      description: t('auth.login.successDesc'),
      color: 'green',
      icon: 'i-heroicons-check-circle'
    })

    await router.push('/dashboard')
  } catch (error: any) {
    console.error('Login failed:', error)

    toast.add({
      title: t('auth.login.error'),
      description: error.message || t('auth.login.errorDesc'),
      color: 'red',
      icon: 'i-heroicons-x-circle'
    })
  } finally {
    isLoading.value = false
  }
}

/**
 * Fill demo credentials
 */
function fillDemoCredentials(role: 'admin' | 'engineer') {
  if (role === 'admin') {
    state.email = 'admin@hydraulic.ai'
    state.password = 'admin123'
  } else {
    state.email = 'engineer@hydraulic.ai'
    state.password = 'eng123'
  }
}
</script>

<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 px-4">
    <div class="w-full max-w-md">
      <!-- Card Container -->
      <UCard class="p-8">
        <!-- Header -->
        <div class="text-center mb-8">
          <div class="mx-auto w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mb-4">
            <UIcon name="i-heroicons-cpu-chip" class="w-6 h-6 text-white" />
          </div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {{ $t('auth.login.title', 'Sign In') }}
          </h1>
          <p class="text-gray-600 dark:text-gray-400 mt-2">
            {{ $t('auth.login.subtitle', 'Hydraulic System Diagnostics') }}
          </p>
        </div>

        <!-- Error Alert -->
        <UAlert
          v-if="authStore.error"
          color="red"
          variant="soft"
          :title="$t('auth.login.error', 'Login Failed')"
          :description="authStore.error"
          icon="i-heroicons-exclamation-triangle"
          class="mb-6"
        />

        <!-- Login Form -->
        <UForm :state="state" @submit="onSubmit" class="space-y-6">
          <!-- Email Field -->
          <UFormGroup
            :label="$t('auth.fields.email', 'Email Address')"
            name="email"
            required
          >
            <UInput
              v-model="state.email"
              type="email"
              :placeholder="$t('auth.placeholders.email', 'user@company.com')"
              icon="i-heroicons-envelope"
              size="lg"
              :disabled="isLoading"
              required
            />
          </UFormGroup>

          <!-- Password Field -->
          <UFormGroup
            :label="$t('auth.fields.password', 'Password')"
            name="password"
            required
          >
            <UInput
              v-model="state.password"
              :type="showPassword ? 'text' : 'password'"
              :placeholder="$t('auth.placeholders.password', 'Enter your password')"
              icon="i-heroicons-lock-closed"
              size="lg"
              :disabled="isLoading"
              required
            >
              <template #trailing>
                <UButton
                  color="gray"
                  variant="link"
                  :icon="showPassword ? 'i-heroicons-eye-slash' : 'i-heroicons-eye'"
                  :padded="false"
                  @click="showPassword = !showPassword"
                />
              </template>
            </UInput>
          </UFormGroup>

          <!-- Submit Button -->
          <UButton
            type="submit"
            color="primary"
            size="lg"
            block
            :loading="isLoading"
            :disabled="!state.email || !state.password"
          >
            {{ $t('auth.login.submit', 'Sign In') }}
          </UButton>
        </UForm>

        <!-- Demo Credentials -->
        <UCard class="mt-6 bg-gray-50 dark:bg-gray-900">
          <div class="space-y-3">
            <p class="text-sm font-medium text-gray-700 dark:text-gray-300">
              {{ $t('auth.demo.title', 'Demo Credentials:') }}
            </p>
            <div class="space-y-2">
              <UButton
                color="white"
                variant="solid"
                size="sm"
                block
                @click="fillDemoCredentials('admin')"
              >
                <div class="flex items-center justify-between w-full text-xs">
                  <span class="font-mono">admin@hydraulic.ai</span>
                  <span class="text-gray-500">/</span>
                  <span class="font-mono">admin123</span>
                </div>
              </UButton>
              <UButton
                color="white"
                variant="solid"
                size="sm"
                block
                @click="fillDemoCredentials('engineer')"
              >
                <div class="flex items-center justify-between w-full text-xs">
                  <span class="font-mono">engineer@hydraulic.ai</span>
                  <span class="text-gray-500">/</span>
                  <span class="font-mono">eng123</span>
                </div>
              </UButton>
            </div>
          </div>
        </UCard>

        <!-- Register Link -->
        <div class="mt-6 text-center">
          <p class="text-sm text-gray-600 dark:text-gray-400">
            {{ $t('auth.login.noAccount', "Don't have an account?") }}
            <NuxtLink
              to="/auth/register"
              class="font-medium text-primary hover:underline"
            >
              {{ $t('auth.login.register', 'Sign Up') }}
            </NuxtLink>
          </p>
        </div>
      </UCard>
    </div>
  </div>
</template>
