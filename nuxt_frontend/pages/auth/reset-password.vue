<script setup lang="ts">
// Password reset confirmation page
definePageMeta({
  layout: 'auth',
  middleware: 'guest'
})

useSeoMeta({
  title: 'Смена пароля | Hydraulic Diagnostic SaaS',
  description: 'Create a new secure password for your hydraulic diagnostics platform account.',
  robots: 'noindex, nofollow'
})

const route = useRoute()
const router = useRouter()

// Get token from URL
const token = route.query.token as string
const email = route.query.email as string

if (!token) {
  // Redirect to forgot password if no token
  navigateTo('/auth/forgot-password')
}

const form = reactive({
  password: '',
  confirmPassword: ''
})

const isLoading = ref(false)
const error = ref('')
const isSuccess = ref(false)
const showPassword = ref(false)
const showConfirmPassword = ref(false)

// Password strength
const passwordStrength = computed(() => {
  const password = form.password
  if (!password) return { score: 0, label: '', color: '' }
  
  let score = 0
  if (password.length >= 8) score += 1
  if (/[A-Z]/.test(password)) score += 1
  if (/[a-z]/.test(password)) score += 1
  if (/\d/.test(password)) score += 1
  if (/[^\w\s]/.test(password)) score += 1
  
  const levels = [
    { score: 0, label: '', color: '' },
    { score: 1, label: 'Очень слабый', color: 'red' },
    { score: 2, label: 'Слабый', color: 'red' },
    { score: 3, label: 'Средний', color: 'yellow' },
    { score: 4, label: 'Сильный', color: 'green' },
    { score: 5, label: 'Очень сильный', color: 'green' }
  ]
  
  return levels[score] || levels[0]
})

const isFormValid = computed(() => {
  return form.password && 
         form.confirmPassword && 
         form.password === form.confirmPassword && 
         form.password.length >= 8 &&
         passwordStrength.value.score >= 3
})

const handlePasswordReset = async () => {
  if (!isFormValid.value) {
    error.value = 'Проверьте корректность заполненных полей'
    return
  }

  isLoading.value = true
  error.value = ''

  try {
    // Call password reset confirmation API
    // await authStore.confirmPasswordReset({
    //   token: token,
    //   password: form.password
    // })
    
    // For demo - simulate success
    await new Promise(resolve => setTimeout(resolve, 2000))
    isSuccess.value = true
    
  } catch (err: any) {
    console.error('Password reset error:', err)
    error.value = err.message || 'Ошибка смены пароля. Ссылка могла устареть.'
  } finally {
    isLoading.value = false
  }
}

const passwordInput = ref<HTMLInputElement>()

onMounted(() => {
  passwordInput.value?.focus()
})
</script>

<template>
  <div class="min-h-screen flex items-center justify-center">
    <div class="max-w-md w-full space-y-8 px-4">
      <!-- Logo and title -->
      <div class="text-center">
        <div class="w-20 h-20 bg-gradient-to-br from-green-600 to-blue-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl">
          <Icon name="heroicons:lock-closed" class="w-10 h-10 text-white" />
        </div>
        
        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Смена пароля
        </h1>
        <p class="text-gray-600 dark:text-gray-300">
          Создайте новый надёжный пароль для вашего аккаунта
        </p>
        <p v-if="email" class="text-sm text-blue-600 dark:text-blue-400 mt-2">
          {{ email }}
        </p>
      </div>

      <!-- Success state -->
      <div v-if="isSuccess" class="text-center space-y-6">
        <div class="w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto">
          <Icon name="heroicons:check-circle" class="w-8 h-8 text-green-600 dark:text-green-400" />
        </div>
        
        <div>
          <h2 class="text-xl font-bold text-gray-900 dark:text-white mb-2">
            Пароль успешно изменён!
          </h2>
          <p class="text-gray-600 dark:text-gray-300 mb-6">
            Теперь вы можете войти в систему с новым паролем
          </p>
        </div>
        
        <NuxtLink
          to="/auth/login"
          class="w-full inline-flex justify-center py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
        >
          <Icon name="heroicons:arrow-right-on-rectangle" class="w-5 h-5 mr-2" />
          Войти в систему
        </NuxtLink>
      </div>

      <!-- Reset form -->
      <form v-else @submit.prevent="handlePasswordReset" class="space-y-6">
        <!-- Error message -->
        <div v-if="error" class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div class="flex items-center space-x-3">
            <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-red-600 dark:text-red-400" />
            <p class="text-sm text-red-700 dark:text-red-300">{{ error }}</p>
          </div>
        </div>

        <!-- New password -->
        <div>
          <label for="password" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
            Новый пароль
          </label>
          <div class="relative">
            <input
              id="password"
              ref="passwordInput"
              v-model="form.password"
              :type="showPassword ? 'text' : 'password'"
              autocomplete="new-password"
              required
              :disabled="isLoading"
              class="w-full px-4 py-3 pr-12 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              placeholder="Минимум 8 символов"
            />
            <button
              type="button"
              @click="showPassword = !showPassword"
              class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <Icon :name="showPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
            </button>
          </div>
          
          <!-- Password strength indicator -->
          <div v-if="form.password" class="mt-2">
            <div class="flex items-center space-x-2">
              <div class="flex-1 h-2 bg-gray-200 dark:bg-gray-700 rounded">
                <div 
                  :class="[
                    'h-2 rounded transition-all duration-300',
                    passwordStrength.color === 'red' ? 'bg-red-500' :
                    passwordStrength.color === 'yellow' ? 'bg-yellow-500' :
                    passwordStrength.color === 'green' ? 'bg-green-500' : 'bg-gray-300'
                  ]"
                  :style="`width: ${(passwordStrength.score / 5) * 100}%`"
                ></div>
              </div>
              <span :class="[
                'text-xs font-medium',
                passwordStrength.color === 'red' ? 'text-red-600 dark:text-red-400' :
                passwordStrength.color === 'yellow' ? 'text-yellow-600 dark:text-yellow-400' :
                passwordStrength.color === 'green' ? 'text-green-600 dark:text-green-400' : 'text-gray-500'
              ]">
                {{ passwordStrength.label }}
              </span>
            </div>
            
            <!-- Password requirements -->
            <div class="mt-3 text-xs space-y-1">
              <div :class="[
                'flex items-center space-x-2',
                form.password.length >= 8 ? 'text-green-600 dark:text-green-400' : 'text-gray-500 dark:text-gray-400'
              ]">
                <Icon :name="form.password.length >= 8 ? 'heroicons:check-circle' : 'heroicons:x-circle'" class="w-3 h-3" />
                <span>Минимум 8 символов</span>
              </div>
              <div :class="[
                'flex items-center space-x-2',
                /[A-Z]/.test(form.password) ? 'text-green-600 dark:text-green-400' : 'text-gray-500 dark:text-gray-400'
              ]">
                <Icon :name="/[A-Z]/.test(form.password) ? 'heroicons:check-circle' : 'heroicons:x-circle'" class="w-3 h-3" />
                <span>Заглавная буква</span>
              </div>
              <div :class="[
                'flex items-center space-x-2',
                /\d/.test(form.password) ? 'text-green-600 dark:text-green-400' : 'text-gray-500 dark:text-gray-400'
              ]">
                <Icon :name="/\d/.test(form.password) ? 'heroicons:check-circle' : 'heroicons:x-circle'" class="w-3 h-3" />
                <span>Цифра</span>
              </div>
              <div :class="[
                'flex items-center space-x-2',
                /[^\w\s]/.test(form.password) ? 'text-green-600 dark:text-green-400' : 'text-gray-500 dark:text-gray-400'
              ]">
                <Icon :name="/[^\w\s]/.test(form.password) ? 'heroicons:check-circle' : 'heroicons:x-circle'" class="w-3 h-3" />
                <span>Специальный символ (!@#$%)</span>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Confirm password -->
        <div>
          <label for="confirmPassword" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
            Подтвердите пароль
          </label>
          <div class="relative">
            <input
              id="confirmPassword"
              v-model="form.confirmPassword"
              :type="showConfirmPassword ? 'text' : 'password'"
              autocomplete="new-password"
              required
              :disabled="isLoading"
              :class="[
                'w-full px-4 py-3 pr-12 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50',
                form.confirmPassword && form.password !== form.confirmPassword ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600',
                'bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400'
              ]"
              placeholder="Повторите новый пароль"
            />
            <button
              type="button"
              @click="showConfirmPassword = !showConfirmPassword"
              class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            >
              <Icon :name="showConfirmPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
            </button>
          </div>
          
          <!-- Password match indicator -->
          <div v-if="form.confirmPassword" class="mt-2">
            <div :class="[
              'flex items-center space-x-2 text-xs',
              form.password === form.confirmPassword ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'
            ]">
              <Icon :name="form.password === form.confirmPassword ? 'heroicons:check-circle' : 'heroicons:x-circle'" class="w-3 h-3" />
              <span>{{ form.password === form.confirmPassword ? 'Пароли совпадают' : 'Пароли не совпадают' }}</span>
            </div>
          </div>
        </div>

        <!-- Submit button -->
        <button
          type="submit"
          :disabled="!isFormValid || isLoading"
          class="w-full flex justify-center py-3 px-4 border border-transparent text-base font-bold rounded-lg text-white bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105"
        >
          <span v-if="!isLoading" class="flex items-center">
            <Icon name="heroicons:shield-check" class="w-5 h-5 mr-2" />
            Сохранить новый пароль
          </span>
          
          <span v-else class="flex items-center">
            <Icon name="heroicons:arrow-path" class="w-5 h-5 mr-2 animate-spin" />
            Сохраняем...
          </span>
        </button>
      </form>

      <!-- Security info -->
      <div class="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <div class="flex items-start space-x-3">
          <Icon name="heroicons:information-circle" class="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5" />
          <div>
            <h3 class="text-sm font-semibold text-blue-900 dark:text-blue-100 mb-1">
              Безопасность пароля
            </h3>
            <p class="text-xs text-blue-700 dark:text-blue-300 leading-relaxed">
              Используйте сложный пароль для защиты ваших критически важных производственных данных. Рекомендуем использовать менеджер паролей.
            </p>
          </div>
        </div>
      </div>

      <!-- Back to login -->
      <div class="text-center">
        <NuxtLink
          to="/auth/login"
          class="inline-flex items-center space-x-2 text-sm font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 transition-colors"
        >
          <Icon name="heroicons:arrow-left" class="w-4 h-4" />
          <span>Вернуться ко входу</span>
        </NuxtLink>
      </div>
    </div>
  </div>
</template>