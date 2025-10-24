<script setup lang="ts">
// Professional enterprise registration page
definePageMeta({
  layout: 'auth',
  middleware: 'guest'
})

useSeoMeta({
  title: 'Регистрация | Hydraulic Diagnostic SaaS',
  description: 'Register for enterprise hydraulic systems monitoring. Start your free trial with full platform access and dedicated technical support.',
  robots: 'noindex, nofollow'
})

interface RegisterForm {
  firstName: string
  lastName: string
  email: string
  password: string
  confirmPassword: string
  company: string
  jobTitle: string
  phone: string
  agreeToTerms: boolean
  subscribeUpdates: boolean
}

const authStore = useAuthStore()
const router = useRouter()

const form = reactive<RegisterForm>({
  firstName: '',
  lastName: '',
  email: '',
  password: '',
  confirmPassword: '',
  company: '',
  jobTitle: '',
  phone: '',
  agreeToTerms: false,
  subscribeUpdates: true
})

const isLoading = ref(false)
const error = ref('')
const showPassword = ref(false)
const showConfirmPassword = ref(false)
const currentStep = ref(1) // Multi-step form

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

// Validation
const validation = computed(() => {
  const errors: Record<string, string> = {}
  
  if (form.firstName && form.firstName.length < 2) {
    errors.firstName = 'Имя должно содержать минимум 2 символа'
  }
  
  if (form.lastName && form.lastName.length < 2) {
    errors.lastName = 'Фамилия должна содержать минимум 2 символа'
  }
  
  if (form.email && !form.email.includes('@')) {
    errors.email = 'Введите корректный email адрес'
  }
  
  if (form.password && form.password.length < 8) {
    errors.password = 'Пароль должен содержать минимум 8 символов'
  }
  
  if (form.confirmPassword && form.password !== form.confirmPassword) {
    errors.confirmPassword = 'Пароли не совпадают'
  }
  
  if (form.company && form.company.length < 2) {
    errors.company = 'Название компании обязательно'
  }
  
  return errors
})

// Step validation
const isStep1Valid = computed(() => {
  return form.firstName && form.lastName && form.email && 
         !validation.value.firstName && !validation.value.lastName && !validation.value.email
})

const isStep2Valid = computed(() => {
  return form.password && form.confirmPassword && form.company &&
         passwordStrength.value.score >= 3 && !validation.value.password && 
         !validation.value.confirmPassword && !validation.value.company
})

const isFormComplete = computed(() => {
  return isStep1Valid.value && isStep2Valid.value && form.agreeToTerms
})

// Registration steps
const nextStep = () => {
  if (currentStep.value === 1 && isStep1Valid.value) {
    currentStep.value = 2
  }
}

const previousStep = () => {
  if (currentStep.value === 2) {
    currentStep.value = 1
  }
}

// Submit handler
const handleRegister = async () => {
  if (!isFormComplete.value) {
    error.value = 'Пожалуйста, заполните все обязательные поля и примите условия'
    return
  }

  isLoading.value = true
  error.value = ''

  try {
    await authStore.register({
      first_name: form.firstName,
      last_name: form.lastName,
      email: form.email,
      password: form.password,
      company: form.company,
      job_title: form.jobTitle,
      phone: form.phone,
      subscribe_updates: form.subscribeUpdates
    })
    
    // Success - redirect to dashboard
    await navigateTo('/dashboard')
    
  } catch (err: any) {
    console.error('Registration error:', err)
    error.value = err.message || 'Ошибка регистрации. Попробуйте позже или обратитесь в поддержку.'
  } finally {
    isLoading.value = false
  }
}

// Auto-focus first input
const firstNameInput = ref<HTMLInputElement>()

onMounted(() => {
  firstNameInput.value?.focus()
})
</script>

<template>
  <div class="min-h-screen flex">
    <!-- Left side: Registration form -->
    <div class="flex-1 flex flex-col justify-center px-4 sm:px-6 lg:flex-none lg:px-20 xl:px-24">
      <div class="mx-auto w-full max-w-md lg:w-[420px]">
        <!-- Logo and title -->
        <div class="text-center mb-8">
          <div class="w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl">
            <Icon name="heroicons:chart-bar-square" class="w-10 h-10 text-white" />
          </div>
          
          <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Создайте аккаунт
          </h1>
          <p class="text-gray-600 dark:text-gray-300 text-lg">
            Получите доступ к платформе мониторинга
          </p>
        </div>

        <!-- Progress indicator -->
        <div class="mb-8">
          <div class="flex items-center justify-between mb-4">
            <div class="flex items-center space-x-2">
              <div :class="[
                'w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors',
                currentStep >= 1 ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
              ]">
                1
              </div>
              <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Личные данные</span>
            </div>
            
            <div class="flex-1 mx-4">
              <div class="h-1 bg-gray-200 dark:bg-gray-700 rounded">
                <div :class="[
                  'h-1 rounded transition-all duration-500',
                  currentStep >= 2 ? 'w-full bg-blue-600' : 'w-0 bg-blue-600'
                ]"></div>
              </div>
            </div>
            
            <div class="flex items-center space-x-2">
              <div :class="[
                'w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold transition-colors',
                currentStep >= 2 ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400'
              ]">
                2
              </div>
              <span class="text-sm font-medium text-gray-700 dark:text-gray-300">Компания</span>
            </div>
          </div>
        </div>

        <!-- Error message -->
        <div v-if="error" class="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <div class="flex items-center space-x-3">
            <Icon name="heroicons:exclamation-triangle" class="w-5 h-5 text-red-600 dark:text-red-400" />
            <p class="text-sm text-red-700 dark:text-red-300">{{ error }}</p>
          </div>
        </div>

        <!-- Registration form -->
        <form @submit.prevent="currentStep === 2 ? handleRegister() : nextStep()" class="space-y-6">
          
          <!-- Step 1: Personal Information -->
          <div v-if="currentStep === 1" class="space-y-6">
            <!-- Name fields -->
            <div class="grid grid-cols-2 gap-4">
              <div>
                <label for="firstName" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                  Имя *
                </label>
                <input
                  id="firstName"
                  ref="firstNameInput"
                  v-model="form.firstName"
                  type="text"
                  autocomplete="given-name"
                  required
                  :disabled="isLoading"
                  :class="[
                    'w-full px-3 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50',
                    validation.firstName ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600',
                    'bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400'
                  ]"
                  placeholder="Александр"
                />
                <p v-if="validation.firstName" class="mt-1 text-xs text-red-600 dark:text-red-400">{{ validation.firstName }}</p>
              </div>
              
              <div>
                <label for="lastName" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                  Фамилия *
                </label>
                <input
                  id="lastName"
                  v-model="form.lastName"
                  type="text"
                  autocomplete="family-name"
                  required
                  :disabled="isLoading"
                  :class="[
                    'w-full px-3 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50',
                    validation.lastName ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600',
                    'bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400'
                  ]"
                  placeholder="Плотников"
                />
                <p v-if="validation.lastName" class="mt-1 text-xs text-red-600 dark:text-red-400">{{ validation.lastName }}</p>
              </div>
            </div>

            <!-- Email field -->
            <div>
              <label for="email" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                Электронная почта *
              </label>
              <div class="relative">
                <input
                  id="email"
                  v-model="form.email"
                  type="email"
                  autocomplete="email"
                  required
                  :disabled="isLoading"
                  :class="[
                    'w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50',
                    validation.email ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600',
                    'bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400'
                  ]"
                  placeholder="aleksandr.plotnikov@company.ru"
                />
                <Icon 
                  name="heroicons:at-symbol" 
                  class="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" 
                />
              </div>
              <p v-if="validation.email" class="mt-1 text-xs text-red-600 dark:text-red-400">{{ validation.email }}</p>
            </div>
            
            <!-- Phone field -->
            <div>
              <label for="phone" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                Номер телефона
                <span class="text-gray-500 text-xs font-normal">(опционально)</span>
              </label>
              <input
                id="phone"
                v-model="form.phone"
                type="tel"
                autocomplete="tel"
                :disabled="isLoading"
                class="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                placeholder="+7 (999) 123-45-67"
              />
            </div>
          </div>
          
          <!-- Step 2: Company & Security -->
          <div v-if="currentStep === 2" class="space-y-6">
            <!-- Company info -->
            <div class="grid grid-cols-1 gap-4">
              <div>
                <label for="company" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                  Компания *
                </label>
                <input
                  id="company"
                  v-model="form.company"
                  type="text"
                  autocomplete="organization"
                  required
                  :disabled="isLoading"
                  :class="[
                    'w-full px-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50',
                    validation.company ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600',
                    'bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400'
                  ]"
                  placeholder="ООО 'Промышленные системы'"
                />
                <p v-if="validation.company" class="mt-1 text-xs text-red-600 dark:text-red-400">{{ validation.company }}</p>
              </div>
              
              <div>
                <label for="jobTitle" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                  Должность
                  <span class="text-gray-500 text-xs font-normal">(опционально)</span>
                </label>
                <input
                  id="jobTitle"
                  v-model="form.jobTitle"
                  type="text"
                  autocomplete="organization-title"
                  :disabled="isLoading"
                  class="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                  placeholder="Начальник производства"
                />
              </div>
            </div>

            <!-- Password fields -->
            <div class="grid grid-cols-1 gap-4">
              <!-- Password -->
              <div>
                <label for="password" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                  Пароль *
                </label>
                <div class="relative">
                  <input
                    id="password"
                    v-model="form.password"
                    :type="showPassword ? 'text' : 'password'"
                    autocomplete="new-password"
                    required
                    :disabled="isLoading"
                    :class="[
                      'w-full px-4 py-3 pr-12 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50',
                      validation.password ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600',
                      'bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400'
                    ]"
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
                </div>
                
                <p v-if="validation.password" class="mt-1 text-xs text-red-600 dark:text-red-400">{{ validation.password }}</p>
              </div>
              
              <!-- Confirm Password -->
              <div>
                <label for="confirmPassword" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
                  Повторите пароль *
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
                      validation.confirmPassword ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600',
                      'bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400'
                    ]"
                    placeholder="Повторите пароль"
                  />
                  <button
                    type="button"
                    @click="showConfirmPassword = !showConfirmPassword"
                    class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                  >
                    <Icon :name="showConfirmPassword ? 'heroicons:eye-slash' : 'heroicons:eye'" class="w-5 h-5" />
                  </button>
                </div>
                <p v-if="validation.confirmPassword" class="mt-1 text-xs text-red-600 dark:text-red-400">{{ validation.confirmPassword }}</p>
              </div>
            </div>
          </div>

          <!-- Step navigation -->
          <div class="flex items-center space-x-4">
            <button
              v-if="currentStep === 2"
              type="button"
              @click="previousStep"
              :disabled="isLoading"
              class="flex-1 py-3 px-4 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 font-medium rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors disabled:opacity-50"
            >
              <Icon name="heroicons:arrow-left" class="w-5 h-5 mr-2 inline" />
              Назад
            </button>
            
            <button
              type="submit"
              :disabled="(currentStep === 1 && !isStep1Valid) || (currentStep === 2 && (!isFormComplete || isLoading))"
              class="flex-1 group relative flex justify-center py-3 px-4 border border-transparent text-base font-bold rounded-lg text-white bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105"
            >
              <span v-if="currentStep === 1" class="flex items-center">
                Продолжить
                <Icon name="heroicons:arrow-right" class="w-5 h-5 ml-2" />
              </span>
              
              <span v-else-if="currentStep === 2 && !isLoading" class="flex items-center">
                <Icon name="heroicons:user-plus" class="w-5 h-5 mr-2" />
                Создать аккаунт
              </span>
              
              <span v-else class="flex items-center">
                <Icon name="heroicons:arrow-path" class="w-5 h-5 mr-2 animate-spin" />
                Создание...
              </span>
            </button>
          </div>
          
          <!-- Terms and privacy -->
          <div v-if="currentStep === 2" class="space-y-4">
            <div class="flex items-start space-x-3">
              <input
                id="agreeToTerms"
                v-model="form.agreeToTerms"
                type="checkbox"
                required
                :disabled="isLoading"
                class="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded disabled:opacity-50"
              />
              <div class="text-sm">
                <label for="agreeToTerms" class="text-gray-700 dark:text-gray-300">
                  Я соглашаюсь с 
                  <a href="#" class="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300">
                    условиями сервиса
                  </a> и 
                  <a href="#" class="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300">
                    политикой конфиденциальности
                  </a>
                </label>
              </div>
            </div>
            
            <div class="flex items-start space-x-3">
              <input
                id="subscribeUpdates"
                v-model="form.subscribeUpdates"
                type="checkbox"
                :disabled="isLoading"
                class="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded disabled:opacity-50"
              />
              <label for="subscribeUpdates" class="text-sm text-gray-700 dark:text-gray-300">
                Получать обновления продукта и новости компании
              </label>
            </div>
          </div>
        </form>

        <!-- Login link -->
        <div class="mt-8 text-center">
          <p class="text-sm text-gray-600 dark:text-gray-300">
            Уже есть аккаунт?
            <NuxtLink
              to="/auth/login"
              class="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 transition-colors ml-1"
            >
              Войти
            </NuxtLink>
          </p>
        </div>
      </div>
    </div>

    <!-- Right side: Benefits and trust signals -->
    <div class="hidden lg:block relative flex-1">
      <div class="absolute inset-0 bg-gradient-to-br from-green-600 via-teal-700 to-blue-800">
        <!-- Animated background -->
        <div class="absolute inset-0">
          <div class="absolute top-32 right-32 w-40 h-40 bg-white/10 rounded-full blur-xl animate-pulse"></div>
          <div class="absolute bottom-20 left-20 w-56 h-56 bg-white/5 rounded-full blur-2xl animate-pulse animation-delay-1000"></div>
          <div class="absolute top-20 left-1/2 w-28 h-28 bg-white/10 rounded-full blur-lg animate-pulse animation-delay-500"></div>
        </div>
        
        <!-- Content -->
        <div class="relative h-full flex items-center justify-center p-12">
          <div class="text-center text-white max-w-lg">
            <Icon name="heroicons:rocket-launch" class="w-20 h-20 mx-auto mb-8 text-green-200" />
            
            <h2 class="text-4xl font-bold mb-6">
              Начните экономить сегодня
            </h2>
            
            <p class="text-green-100 text-xl leading-relaxed mb-8">
              Присоединяйтесь к 127+ предприятиям, которые сэкономили миллионы рублей на обслуживании гидравлики.
            </p>
            
            <!-- Benefits list -->
            <div class="space-y-4 text-left">
              <div class="flex items-start space-x-4">
                <div class="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Icon name="heroicons:check" class="w-5 h-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-green-100 mb-1">Бесплатный 14-дневный пробный период</h3>
                  <p class="text-green-200 text-sm">Полный доступ к всем возможностям платформы</p>
                </div>
              </div>
              
              <div class="flex items-start space-x-4">
                <div class="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Icon name="heroicons:check" class="w-5 h-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-green-100 mb-1">Личный менеджер и обучение</h3>
                  <p class="text-green-200 text-sm">Поможем настроить систему под ваши потребности</p>
                </div>
              </div>
              
              <div class="flex items-start space-x-4">
                <div class="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                  <Icon name="heroicons:check" class="w-5 h-5" />
                </div>
                <div>
                  <h3 class="font-semibold text-green-100 mb-1">ROI гарантия 6 месяцев</h3>
                  <p class="text-green-200 text-sm">Окупаемость в первое полугодие или возврат средств</p>
                </div>
              </div>
            </div>
            
            <!-- Contact info -->
            <div class="mt-8 p-4 bg-white/10 backdrop-blur rounded-lg">
              <p class="text-green-100 text-sm mb-2">
                Нужна помощь с регистрацией?
              </p>
              <div class="flex items-center justify-center space-x-4">
                <a href="tel:+74959847621" class="flex items-center space-x-2 text-green-200 hover:text-white transition-colors">
                  <Icon name="heroicons:phone" class="w-4 h-4" />
                  <span class="text-sm font-medium">+7 (495) 984-76-21</span>
                </a>
                <span class="text-green-300">•</span>
                <a href="mailto:support@hydraulic-diagnostics.com" class="flex items-center space-x-2 text-green-200 hover:text-white transition-colors">
                  <Icon name="heroicons:envelope" class="w-4 h-4" />
                  <span class="text-sm font-medium">Поддержка</span>
                </a>
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