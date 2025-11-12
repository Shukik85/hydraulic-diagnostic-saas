/**
 * Register Page - New user registration
 * 
 * Features:
 * - Nuxt UI form components
 * - Dark mode support
 * - i18n ready
 * - Password strength indicator
 * - Password match validation
 * - Terms agreement
 * - Loading states
 * - Error handling
 */
<script setup lang="ts">
import type { FormSubmitEvent } from '#ui/types'
import type { RegisterData } from '~/types/api'

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
interface RegisterForm {
  first_name: string
  last_name: string
  email: string
  password: string
  password_confirm: string
  agreeTerms: boolean
}

const state = reactive<RegisterForm>({
  first_name: '',
  last_name: '',
  email: '',
  password: '',
  password_confirm: '',
  agreeTerms: false
})

const isLoading = ref(false)
const showPassword = ref(false)
const showConfirmPassword = ref(false)

// Password validation
const passwordStrength = computed(() => {
  const password = state.password
  if (!password) return { score: 0, label: '', color: 'gray' }

  let score = 0
  let label = ''
  let color = 'red'

  // Length check
  if (password.length >= 8) score += 25
  if (password.length >= 12) score += 25

  // Character variety
  if (/[a-z]/.test(password)) score += 12.5
  if (/[A-Z]/.test(password)) score += 12.5
  if (/[0-9]/.test(password)) score += 12.5
  if (/[^a-zA-Z0-9]/.test(password)) score += 12.5

  // Determine label and color
  if (score < 40) {
    label = t('auth.password.weak', 'Weak')
    color = 'red'
  } else if (score < 60) {
    label = t('auth.password.fair', 'Fair')
    color = 'yellow'
  } else if (score < 80) {
    label = t('auth.password.good', 'Good')
    color = 'green'
  } else {
    label = t('auth.password.strong', 'Strong')
    color = 'blue'
  }

  return { score: Math.min(score, 100), label, color }
})

const passwordMatch = computed(() => {
  if (!state.password_confirm) return true
  return state.password === state.password_confirm
})

/**
 * Handle registration form submission
 */
async function onSubmit(event: FormSubmitEvent<RegisterForm>) {
  if (!passwordMatch.value) {
    toast.add({
      title: t('auth.register.passwordMismatch'),
      color: 'red',
      icon: 'i-heroicons-x-circle'
    })
    return
  }

  if (!state.agreeTerms) {
    toast.add({
      title: t('auth.register.agreeTermsRequired'),
      color: 'red',
      icon: 'i-heroicons-x-circle'
    })
    return
  }

  isLoading.value = true

  try {
    const registerData: RegisterData = {
      email: event.data.email,
      password: event.data.password,
      password_confirm: event.data.password_confirm,
      first_name: event.data.first_name,
      last_name: event.data.last_name
    }

    await authStore.register(registerData)

    toast.add({
      title: t('auth.register.success'),
      description: t('auth.register.successDesc'),
      color: 'green',
      icon: 'i-heroicons-check-circle'
    })

    await router.push('/dashboard')
  } catch (error: any) {
    console.error('Registration failed:', error)

    toast.add({
      title: t('auth.register.error'),
      description: error.message || t('auth.register.errorDesc'),
      color: 'red',
      icon: 'i-heroicons-x-circle'
    })
  } finally {
    isLoading.value = false
  }
}
</script>

<template>
  <div class="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 px-4 py-8">
    <div class="w-full max-w-md">
      <!-- Card Container -->
      <UCard class="p-8">
        <!-- Header -->
        <div class="text-center mb-8">
          <div class="mx-auto w-12 h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mb-4">
            <UIcon name="i-heroicons-user-plus" class="w-6 h-6 text-white" />
          </div>
          <h1 class="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {{ $t('auth.register.title', 'Create Account') }}
          </h1>
          <p class="text-gray-600 dark:text-gray-400 mt-2">
            {{ $t('auth.register.subtitle', 'Get access to the platform') }}
          </p>
        </div>

        <!-- Registration Form -->
        <UForm :state="state" @submit="onSubmit" class="space-y-6">
          <!-- Name Fields -->
          <div class="grid grid-cols-2 gap-4">
            <UFormGroup
              :label="$t('auth.fields.firstName', 'First Name')"
              name="first_name"
            >
              <UInput
                v-model="state.first_name"
                :placeholder="$t('auth.placeholders.firstName', 'John')"
                size="lg"
                :disabled="isLoading"
              />
            </UFormGroup>

            <UFormGroup
              :label="$t('auth.fields.lastName', 'Last Name')"
              name="last_name"
            >
              <UInput
                v-model="state.last_name"
                :placeholder="$t('auth.placeholders.lastName', 'Doe')"
                size="lg"
                :disabled="isLoading"
              />
            </UFormGroup>
          </div>

          <!-- Email Field -->
          <UFormGroup
            :label="$t('auth.fields.email', 'Email Address')"
            name="email"
            required
          >
            <UInput
              v-model="state.email"
              type="email"
              :placeholder="$t('auth.placeholders.email', 'john.doe@company.com')"
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
              :placeholder="$t('auth.placeholders.createPassword', 'Create a strong password')"
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

            <!-- Password Strength Indicator -->
            <div v-if="state.password" class="mt-3 space-y-2">
              <div class="flex items-center justify-between">
                <span class="text-xs text-gray-600 dark:text-gray-400">
                  {{ $t('auth.password.strength', 'Password Strength') }}
                </span>
                <UBadge :color="passwordStrength.color" size="xs">
                  {{ passwordStrength.label }}
                </UBadge>
              </div>
              <UProgress :value="passwordStrength.score" :color="passwordStrength.color" />
            </div>
          </UFormGroup>

          <!-- Confirm Password Field -->
          <UFormGroup
            :label="$t('auth.fields.confirmPassword', 'Confirm Password')"
            name="password_confirm"
            required
          >
            <UInput
              v-model="state.password_confirm"
              :type="showConfirmPassword ? 'text' : 'password'"
              :placeholder="$t('auth.placeholders.confirmPassword', 'Re-enter your password')"
              icon="i-heroicons-lock-closed"
              size="lg"
              :disabled="isLoading"
              required
            >
              <template #trailing>
                <UButton
                  color="gray"
                  variant="link"
                  :icon="showConfirmPassword ? 'i-heroicons-eye-slash' : 'i-heroicons-eye'"
                  :padded="false"
                  @click="showConfirmPassword = !showConfirmPassword"
                />
              </template>
            </UInput>

            <!-- Password Match Indicator -->
            <div v-if="state.password_confirm" class="mt-2">
              <div v-if="!passwordMatch" class="flex items-center gap-2 text-xs text-red-600 dark:text-red-400">
                <UIcon name="i-heroicons-x-circle" class="w-4 h-4" />
                <span>{{ $t('auth.password.mismatch', 'Passwords do not match') }}</span>
              </div>
              <div v-else class="flex items-center gap-2 text-xs text-green-600 dark:text-green-400">
                <UIcon name="i-heroicons-check-circle" class="w-4 h-4" />
                <span>{{ $t('auth.password.match', 'Passwords match') }}</span>
              </div>
            </div>
          </UFormGroup>

          <!-- Terms Agreement -->
          <UFormGroup name="agreeTerms">
            <UCheckbox
              v-model="state.agreeTerms"
              :disabled="isLoading"
              required
            >
              <template #label>
                <span class="text-sm text-gray-600 dark:text-gray-400">
                  {{ $t('auth.register.agreeText', 'I agree to the') }}
                  <NuxtLink to="/terms" class="text-primary hover:underline">
                    {{ $t('auth.register.terms', 'Terms of Service') }}
                  </NuxtLink>
                  {{ $t('ui.and', 'and') }}
                  <NuxtLink to="/privacy" class="text-primary hover:underline">
                    {{ $t('auth.register.privacy', 'Privacy Policy') }}
                  </NuxtLink>
                </span>
              </template>
            </UCheckbox>
          </UFormGroup>

          <!-- Submit Button -->
          <UButton
            type="submit"
            color="primary"
            size="lg"
            block
            :loading="isLoading"
            :disabled="!state.email || !state.password || !passwordMatch || !state.agreeTerms"
          >
            {{ $t('auth.register.submit', 'Create Account') }}
          </UButton>
        </UForm>

        <!-- Login Link -->
        <div class="mt-6 text-center">
          <p class="text-sm text-gray-600 dark:text-gray-400">
            {{ $t('auth.register.haveAccount', 'Already have an account?') }}
            <NuxtLink
              to="/auth/login"
              class="font-medium text-primary hover:underline"
            >
              {{ $t('auth.register.login', 'Sign In') }}
            </NuxtLink>
          </p>
        </div>
      </UCard>
    </div>
  </div>
</template>
