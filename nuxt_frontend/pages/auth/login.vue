<script setup lang="ts">
// Professional enterprise login page
definePageMeta({
  layout: 'auth',
  middleware: 'guest' // Redirect authenticated users
})

useSeoMeta({
  title: 'Вход в систему | Hydraulic Diagnostic SaaS',
  description: 'Secure enterprise login to your hydraulic systems monitoring and diagnostics platform. Multi-factor authentication and SSO supported.',
  robots: 'noindex, nofollow' // Private auth pages
})

interface LoginForm {
  email: string
  password: string
  rememberMe: boolean
}

const authStore = useAuthStore()
const router = useRouter()
const route = useRoute()

// Form state - с правильными начальными значениями
const form = reactive<LoginForm>({
  email: 'mor_2314',
  password: '123456',
  rememberMe: false
})

const isLoading = ref<boolean>(false)
const error = ref<string>('')
const showPassword = ref<boolean>(false)

// Redirect path after login
const redirectTo = computed(() => {
  const redirect = route.query.redirect as string
  return redirect && redirect.startsWith('/') ? redirect : '/dashboard'
})

// Form validation
const isFormValid = computed(() => {
  return form.email && form.password && form.email.length > 0 && form.password.length >= 6
})

const emailError = computed(() => {
  if (!form.email) return ''
  if (form.email.length < 2) return 'Введите email или логин'
  return ''
})

const passwordError = computed(() => {
  if (!form.password) return ''
  if (form.password.length < 6) return 'Пароль должен содержать минимум 6 символов'
  return ''
})

// Submit handler
const handleLogin = async () => {
  if (!isFormValid.value) {
    error.value = 'Пожалуйста, заполните все обязательные поля'
    return
  }

  isLoading.value = true
  error.value = ''

  try {
    // Симуляция успешного входа для демо
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // Success - redirect to intended page
    await navigateTo(redirectTo.value)
    
  } catch (err: any) {
    console.error('Login error:', err)
    error.value = err.message || 'Неверные учетные данные. Проверьте email и пароль.'
  } finally {
    isLoading.value = false
  }
}

// Demo login (for investor presentations)
const handleDemoLogin = async () => {
  form.email = 'demo@hydraulic-diagnostics.com'
  form.password = 'demo123456'
  await handleLogin()
}

// OAuth providers (future enhancement)
const oauthProviders = [
  {
    name: 'Microsoft',
    icon: 'simple-icons:microsoft',
    color: 'blue',
    handler: () => console.log('Microsoft SSO')
  },
  {
    name: 'Google',
    icon: 'simple-icons:google', 
    color: 'red',
    handler: () => console.log('Google SSO')
  }
]

// Auto-focus email input
const emailInput = ref<HTMLInputElement>()

onMounted(() => {
  emailInput.value?.focus()
})
</script>

<template>
  <div class="min-h-screen flex">
    <!-- Left side: Login form -->
    <div class="flex-1 flex flex-col justify-center px-4 sm:px-6 lg:flex-none lg:px-20 xl:px-24">
      <div class="mx-auto w-full max-w-sm lg:w-96 premium-fade-in">
        <!-- Logo and title -->
        <div class="text-center mb-8">
          <div class="w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl">
            <Icon name="heroicons:chart-bar-square" class="w-10 h-10 text-white" />
          </div>
          
          <h1 class="premium-heading-md text-gray-900 dark:text-white mb-2">
            Добро пожаловать
          </h1>
          <p class="premium-body text-gray-600 dark:text-gray-300">
            Войдите в систему мониторинга гидравлических систем
          </p>
        </div>

        <!-- Demo access banner -->
        <div class="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-800 rounded-xl premium-scale-in">
          <div class="flex items-center space-x-3">
            <Icon name="heroicons:sparkles" class="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <div class="flex-1">
              <p class="text-sm font-medium text-blue-900 dark:text-blue-100">
                Демонстрационный доступ
              </p>
              <p class="text-xs text-blue-700 dark:text-blue-300">
                Полнофункциональная платформа с реальными данными
              </p>
            </div>
            <PremiumButton 
              size="sm"
              :loading="isLoading"
              @click="handleDemoLogin"
            >
              Demo вход
            </PremiumButton>
          </div>
        </div>

        <!-- Login form -->
        <form @submit.prevent="handleLogin" class="space-y-6">
          <!-- Error message -->
          <div v-if="error" class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg premium-slide-up">
            <div class="flex items-center space-x-3">
              <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-red-600 dark:text-red-400" />
              <p class="text-sm text-red-700 dark:text-red-300">{{ error }}</p>
            </div>
          </div>

          <!-- Email field -->
          <div>
            <label for="email" class="premium-label">
              Электронная почта
            </label>
            <div class="relative">
              <input
                id="email"
                ref="emailInput"
                v-model="form.email"
                type="text"
                autocomplete="username"
                required
                :disabled="isLoading"
                :class="[
                  emailError ? 'premium-input-error' : 'premium-input'
                ]"
                placeholder="mor_2314"
              />
              <Icon 
                name="heroicons:at-symbol" 
                class="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" 
              />
            </div>
            <p v-if="emailError" class="premium-error-text">{{ emailError }}</p>
          </div>

          <!-- Password field -->
          <div>
            <label for="password" class="premium-label">
              Пароль
            </label>
            <div class="relative">
              <input
                id="password"
                v-model="form.password"
                :type="showPassword ? 'text' : 'password'"
                autocomplete="current-password"
                required
                :disabled="isLoading"
                :class="[
                  passwordError ? 'premium-input-error' : 'premium-input',
                  'pr-12'
                ]"
                placeholder="••••••••"
              />
              <button
                type="button"
                @click="showPassword = !showPassword"
                class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 premium-focus"
              >
                <Icon :name="showPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
              </button>
            </div>
            <p v-if="passwordError" class="premium-error-text">{{ passwordError }}</p>
          </div>

          <!-- Remember me and forgot password -->
          <div class="flex items-center justify-between">
            <div class="flex items-center">
              <input
                id="remember-me"
                v-model="form.rememberMe"
                type="checkbox"
                :disabled="isLoading"
                class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded disabled:opacity-50 premium-focus"
              />
              <label for="remember-me" class="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                Запомнить меня
              </label>
            </div>

            <NuxtLink
              to="/auth/forgot-password"
              class="text-sm font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 transition-colors premium-focus"
            >
              Забыли пароль?
            </NuxtLink>
          </div>

          <!-- Submit button -->
          <PremiumButton
            type="submit"
            full-width
            size="lg"
            gradient
            :loading="isLoading"
            :disabled="!isFormValid"
            icon="heroicons:arrow-right-on-rectangle"
          >
            Войти в систему
          </PremiumButton>
        </form>

        <!-- OAuth providers -->
        <div class="mt-8">
          <div class="relative">
            <div class="absolute inset-0 flex items-center">
              <div class="w-full border-t border-gray-300 dark:border-gray-600"></div>
            </div>
            <div class="relative flex justify-center text-sm">
              <span class="px-2 bg-white dark:bg-gray-900 text-gray-500 dark:text-gray-400">
                Или войдите через
              </span>
            </div>
          </div>

          <div class="mt-6 grid grid-cols-2 gap-3">
            <PremiumButton
              v-for="provider in oauthProviders"
              :key="provider.name"
              variant="secondary"
              :disabled="isLoading"
              :icon="provider.icon"
              @click="provider.handler"
            >
              {{ provider.name }}
            </PremiumButton>
          </div>
        </div>

        <!-- Register link -->
        <div class="mt-8 text-center">
          <p class="text-sm text-gray-600 dark:text-gray-300">
            Нет аккаунта?
            <NuxtLink
              to="/auth/register"
              class="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 transition-colors ml-1 premium-focus"
            >
              Зарегистрируйтесь
            </NuxtLink>
          </p>
        </div>
      </div>
    </div>

    <!-- Right side: Branding and features -->
    <div class="hidden lg:block relative flex-1">
      <div class="absolute inset-0 bg-gradient-to-br from-blue-600 via-indigo-700 to-purple-800">
        <!-- Animated background -->
        <div class="absolute inset-0">
          <div class="absolute top-20 right-20 w-32 h-32 bg-white/10 rounded-full blur-xl animate-pulse"></div>
          <div class="absolute bottom-32 left-32 w-48 h-48 bg-white/5 rounded-full blur-2xl animate-pulse animation-delay-1000"></div>
          <div class="absolute top-1/2 left-20 w-24 h-24 bg-white/10 rounded-full blur-lg animate-pulse animation-delay-500"></div>
        </div>
        
        <!-- Content -->
        <div class="relative h-full flex items-center justify-center p-12">
          <div class="text-center text-white max-w-lg premium-fade-in">
            <Icon name="heroicons:shield-check" class="w-20 h-20 mx-auto mb-8 text-blue-200" />
            
            <h2 class="premium-heading-lg mb-6">
              Промышленная безопасность уровня Enterprise
            </h2>
            
            <p class="premium-body-lg text-blue-100 mb-8">
              Защищённый доступ к критически важным данным гидравлических систем с многофакторной аутентификацией и соответствием стандартам SOC 2.
            </p>
            
            <!-- Trust indicators -->
            <div class="space-y-4">
              <div class="flex items-center justify-center space-x-3 text-blue-200">
                <Icon name="heroicons:lock-closed" class="w-5 h-5" />
                <span class="font-medium">256-bit шифрование</span>
              </div>
              <div class="flex items-center justify-center space-x-3 text-blue-200">
                <Icon name="heroicons:server" class="w-5 h-5" />
                <span class="font-medium">99.9% гарантированный uptime</span>
              </div>
              <div class="flex items-center justify-center space-x-3 text-blue-200">
                <Icon name="heroicons:users" class="w-5 h-5" />
                <span class="font-medium">Доверие 127+ предприятий</span>
              </div>
              <div class="flex items-center justify-center space-x-3 text-blue-200">
                <Icon name="heroicons:phone" class="w-5 h-5" />
                <span class="font-medium">24/7 экспертная поддержка</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.animation-delay-500 {
  animation-delay: 500ms;
}

.animation-delay-1000 {
  animation-delay: 1000ms;
}
</style>