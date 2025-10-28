<script setup lang="ts">
// Enhanced registration with proper TypeScript and null safety
definePageMeta({
  layout: 'auth',
  middleware: 'guest',
});

useSeoMeta({
  title: 'Регистрация | Hydraulic Diagnostic SaaS',
  description:
    'Create your enterprise account for hydraulic systems monitoring and diagnostics platform.',
  robots: 'noindex, nofollow',
});

interface RegisterForm {
  email: string;
  password: string;
  confirmPassword: string;
  firstName: string;
  lastName: string;
  company: string;
  jobTitle: string;
  phone: string;
  subscribeUpdates: boolean;
  termsAccepted: boolean;
}

const authStore = useAuthStore();
const router = useRouter();

// Form state with proper initial values
const currentStep = ref<number>(1);
const form = reactive<RegisterForm>({
  email: '',
  password: '',
  confirmPassword: '',
  firstName: '',
  lastName: '',
  company: '',
  jobTitle: '',
  phone: '',
  subscribeUpdates: true,
  termsAccepted: false,
});

const isLoading = ref<boolean>(false);
const error = ref<string>('');
const showPassword = ref<boolean>(false);
const showConfirmPassword = ref<boolean>(false);

// Password strength with guaranteed non-null return
const passwordStrength = usePasswordStrength(toRef(form, 'password'));

// Form validation with null safety
const validation = computed(() => ({
  email: !form.email || !form.email.includes('@') ? 'Введите корректный email' : '',
  password: !form.password || form.password.length < 8 ? 'Минимум 8 символов' : '',
  confirmPassword: form.password !== form.confirmPassword ? 'Пароли не совпадают' : '',
  firstName: !form.firstName ? 'Обязательное поле' : '',
  lastName: !form.lastName ? 'Обязательное поле' : '',
  company: !form.company ? 'Обязательное поле' : '',
  terms: !form.termsAccepted ? 'Необходимо принять условия' : '',
}));

const isStep1Valid = computed(() => {
  return !validation.value.email && !validation.value.firstName && !validation.value.lastName;
});

const isStep2Valid = computed(() => {
  return (
    form.password &&
    form.confirmPassword &&
    form.company &&
    passwordStrength.value.score >= 3 &&
    !validation.value.password &&
    !validation.value.confirmPassword &&
    !validation.value.company
  );
});

const isStep3Valid = computed(() => {
  return form.termsAccepted && !validation.value.terms;
});

// Navigation
const nextStep = () => {
  if (currentStep.value < 3) {
    currentStep.value++;
  }
};

const prevStep = () => {
  if (currentStep.value > 1) {
    currentStep.value--;
  }
};

// Submit handler
const handleRegister = async () => {
  if (!isStep3Valid.value) return;

  isLoading.value = true;
  error.value = '';

  try {
    await authStore.register({
      email: form.email,
      password: form.password,
      first_name: form.firstName,
      last_name: form.lastName,
      company: form.company,
      job_title: form.jobTitle,
      phone: form.phone,
      subscribe_updates: form.subscribeUpdates,
    });

    await navigateTo('/dashboard');
  } catch (err: any) {
    console.error('Registration error:', err);
    error.value = err.message || 'Ошибка регистрации. Попробуйте позже.';
  } finally {
    isLoading.value = false;
  }
};

// Auto-focus first input
const emailInput = ref<HTMLInputElement>();

onMounted(() => {
  emailInput.value?.focus();
});
</script>

<template>
  <div class="min-h-screen flex">
    <!-- Left side: Registration form -->
    <div class="flex-1 flex flex-col justify-center px-4 sm:px-6 lg:flex-none lg:px-20 xl:px-24">
      <div class="mx-auto w-full max-w-sm lg:w-96 premium-fade-in">
        <!-- Logo and title -->
        <div class="text-center mb-8">
          <div
            class="w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl"
          >
            <Icon name="heroicons:user-plus" class="w-10 h-10 text-white" />
          </div>

          <h1 class="premium-heading-md text-gray-900 dark:text-white mb-2">Создайте аккаунт</h1>
          <p class="premium-body text-gray-600 dark:text-gray-300">
            Получите доступ к платформе мониторинга
          </p>
        </div>

        <!-- Progress indicator -->
        <div class="mb-8">
          <div class="flex items-center justify-between mb-2">
            <span class="text-sm font-medium text-gray-500 dark:text-gray-400"
              >Шаг {{ currentStep }} из 3</span
            >
            <span class="text-sm font-medium text-gray-500 dark:text-gray-400"
              >{{ Math.round((currentStep / 3) * 100) }}%</span
            >
          </div>
          <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <div
              class="bg-blue-600 h-2 rounded-full transition-all duration-300"
              :style="`width: ${(currentStep / 3) * 100}%`"
            ></div>
          </div>
        </div>

        <!-- Error message -->
        <div
          v-if="error"
          class="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg premium-slide-up"
        >
          <div class="flex items-center space-x-3">
            <Icon
              name="heroicons:exclamation-triangle"
              class="w-5 h-5 text-red-600 dark:text-red-400"
            />
            <p class="text-sm text-red-700 dark:text-red-300">{{ error }}</p>
          </div>
        </div>

        <!-- Step 1: Personal Information -->
        <form v-if="currentStep === 1" @submit.prevent="nextStep" class="space-y-6">
          <!-- Email -->
          <div>
            <label for="email" class="premium-label">Электронная почта</label>
            <input
              id="email"
              ref="emailInput"
              v-model="form.email"
              type="email"
              required
              :disabled="isLoading"
              :class="validation.email ? 'premium-input-error' : 'premium-input'"
              placeholder="your.email@company.com"
            />
            <p v-if="validation.email" class="premium-error-text">{{ validation.email }}</p>
          </div>

          <!-- Name fields -->
          <div class="grid grid-cols-2 gap-4">
            <div>
              <label for="firstName" class="premium-label">Имя</label>
              <input
                id="firstName"
                v-model="form.firstName"
                type="text"
                required
                :disabled="isLoading"
                :class="validation.firstName ? 'premium-input-error' : 'premium-input'"
                placeholder="Иван"
              />
              <p v-if="validation.firstName" class="premium-error-text">
                {{ validation.firstName }}
              </p>
            </div>
            <div>
              <label for="lastName" class="premium-label">Фамилия</label>
              <input
                id="lastName"
                v-model="form.lastName"
                type="text"
                required
                :disabled="isLoading"
                :class="validation.lastName ? 'premium-input-error' : 'premium-input'"
                placeholder="Иванов"
              />
              <p v-if="validation.lastName" class="premium-error-text">{{ validation.lastName }}</p>
            </div>
          </div>

          <PremiumButton
            type="submit"
            full-width
            size="lg"
            gradient
            :disabled="!isStep1Valid"
            icon="heroicons:arrow-right"
          >
            Продолжить
          </PremiumButton>
        </form>

        <!-- Step 2: Security -->
        <form v-else-if="currentStep === 2" @submit.prevent="nextStep" class="space-y-6">
          <!-- Password -->
          <div>
            <label for="password" class="premium-label">Пароль</label>
            <div class="relative">
              <input
                id="password"
                v-model="form.password"
                :type="showPassword ? 'text' : 'password'"
                required
                :disabled="isLoading"
                :class="[validation.password ? 'premium-input-error' : 'premium-input', 'pr-12']"
                placeholder="••••••••"
              />
              <button
                type="button"
                @click="showPassword = !showPassword"
                class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <Icon
                  :name="showPassword ? 'heroicons:eye-slash' : 'heroicons:eye'"
                  class="w-5 h-5"
                />
              </button>
            </div>
            <p v-if="validation.password" class="premium-error-text">{{ validation.password }}</p>

            <!-- Password strength indicator - Fixed null safety -->
            <div v-if="form.password" class="mt-2">
              <div class="flex items-center justify-between mb-1">
                <span class="text-xs text-gray-500 dark:text-gray-400">Надёжность пароля</span>
                <span
                  :class="[
                    'text-xs font-medium',
                    passwordStrength.color === 'red'
                      ? 'text-red-600 dark:text-red-400'
                      : passwordStrength.color === 'yellow'
                        ? 'text-yellow-600 dark:text-yellow-400'
                        : passwordStrength.color === 'green'
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-gray-500',
                  ]"
                >
                  {{ passwordStrength.label }}
                </span>
              </div>
              <div class="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div
                  :class="[
                    'h-2 rounded transition-all duration-300',
                    passwordStrength.color === 'red'
                      ? 'bg-red-500'
                      : passwordStrength.color === 'yellow'
                        ? 'bg-yellow-500'
                        : passwordStrength.color === 'green'
                          ? 'bg-green-500'
                          : 'bg-gray-300',
                  ]"
                  :style="`width: ${(passwordStrength.score / 5) * 100}%`"
                ></div>
              </div>
            </div>
          </div>

          <!-- Confirm Password -->
          <div>
            <label for="confirmPassword" class="premium-label">Подтвердите пароль</label>
            <div class="relative">
              <input
                id="confirmPassword"
                v-model="form.confirmPassword"
                :type="showConfirmPassword ? 'text' : 'password'"
                required
                :disabled="isLoading"
                :class="[
                  validation.confirmPassword ? 'premium-input-error' : 'premium-input',
                  'pr-12',
                ]"
                placeholder="••••••••"
              />
              <button
                type="button"
                @click="showConfirmPassword = !showConfirmPassword"
                class="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
              >
                <Icon
                  :name="showConfirmPassword ? 'heroicons:eye-slash' : 'heroicons:eye'"
                  class="w-5 h-5"
                />
              </button>
            </div>
            <p v-if="validation.confirmPassword" class="premium-error-text">
              {{ validation.confirmPassword }}
            </p>
          </div>

          <!-- Company -->
          <div>
            <label for="company" class="premium-label">Компания</label>
            <input
              id="company"
              v-model="form.company"
              type="text"
              required
              :disabled="isLoading"
              :class="validation.company ? 'premium-input-error' : 'premium-input'"
              placeholder="ООО Гидравлик Системс"
            />
            <p v-if="validation.company" class="premium-error-text">{{ validation.company }}</p>
          </div>

          <!-- Navigation buttons -->
          <div class="flex space-x-4">
            <PremiumButton
              type="button"
              variant="secondary"
              size="lg"
              :disabled="isLoading"
              @click="prevStep"
              icon="heroicons:arrow-left"
              class="flex-1"
            >
              Назад
            </PremiumButton>
            <PremiumButton
              type="submit"
              size="lg"
              gradient
              :disabled="!isStep2Valid"
              icon="heroicons:arrow-right"
              class="flex-1"
            >
              Продолжить
            </PremiumButton>
          </div>
        </form>

        <!-- Step 3: Final Details -->
        <form v-else @submit.prevent="handleRegister" class="space-y-6">
          <!-- Optional fields -->
          <div>
            <label for="jobTitle" class="premium-label">Должность (необязательно)</label>
            <input
              id="jobTitle"
              v-model="form.jobTitle"
              type="text"
              :disabled="isLoading"
              class="premium-input"
              placeholder="Начальник отдела обслуживания"
            />
          </div>

          <div>
            <label for="phone" class="premium-label">Телефон (необязательно)</label>
            <input
              id="phone"
              v-model="form.phone"
              type="tel"
              :disabled="isLoading"
              class="premium-input"
              placeholder="+7 (999) 123-45-67"
            />
          </div>

          <!-- Preferences -->
          <div class="space-y-4">
            <div class="flex items-center">
              <input
                id="subscribe"
                v-model="form.subscribeUpdates"
                type="checkbox"
                :disabled="isLoading"
                class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
              />
              <label for="subscribe" class="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                Получать обновления по электронной почте
              </label>
            </div>

            <div class="flex items-start">
              <input
                id="terms"
                v-model="form.termsAccepted"
                type="checkbox"
                required
                :disabled="isLoading"
                class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded mt-1"
              />
              <label for="terms" class="ml-2 block text-sm text-gray-700 dark:text-gray-300">
                Я соглашаюсь с
                <a href="/terms" class="text-blue-600 hover:text-blue-500 dark:text-blue-400"
                  >условиями использования</a
                >
                и
                <a href="/privacy" class="text-blue-600 hover:text-blue-500 dark:text-blue-400"
                  >политикой конфиденциальности</a
                >
              </label>
              <p v-if="validation.terms" class="premium-error-text">{{ validation.terms }}</p>
            </div>
          </div>

          <!-- Navigation buttons -->
          <div class="flex space-x-4">
            <PremiumButton
              type="button"
              variant="secondary"
              size="lg"
              :disabled="isLoading"
              @click="prevStep"
              icon="heroicons:arrow-left"
              class="flex-1"
            >
              Назад
            </PremiumButton>
            <PremiumButton
              type="submit"
              size="lg"
              gradient
              :loading="isLoading"
              :disabled="!isStep3Valid"
              icon="heroicons:user-plus"
              class="flex-1"
            >
              Создать аккаунт
            </PremiumButton>
          </div>
        </form>

        <!-- Login link -->
        <div class="mt-8 text-center">
          <p class="text-sm text-gray-600 dark:text-gray-300">
            Уже есть аккаунт?
            <NuxtLink
              to="/auth/login"
              class="font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 transition-colors ml-1 premium-focus"
            >
              Войдите
            </NuxtLink>
          </p>
        </div>
      </div>
    </div>

    <!-- Right side: Benefits -->
    <div class="hidden lg:block relative flex-1">
      <div class="absolute inset-0 bg-gradient-to-br from-purple-600 via-indigo-700 to-blue-800">
        <!-- Animated background -->
        <div class="absolute inset-0">
          <div
            class="absolute top-20 right-20 w-32 h-32 bg-white/10 rounded-full blur-xl animate-pulse"
          ></div>
          <div
            class="absolute bottom-32 left-32 w-48 h-48 bg-white/5 rounded-full blur-2xl animate-pulse animation-delay-1000"
          ></div>
          <div
            class="absolute top-1/2 left-20 w-24 h-24 bg-white/10 rounded-full blur-lg animate-pulse animation-delay-500"
          ></div>
        </div>

        <!-- Content -->
        <div class="relative h-full flex items-center justify-center p-12">
          <div class="text-center text-white max-w-lg premium-fade-in">
            <Icon name="heroicons:rocket-launch" class="w-20 h-20 mx-auto mb-8 text-purple-200" />

            <h2 class="premium-heading-lg mb-6">Присоединяйтесь к инновациям</h2>

            <p class="premium-body-lg text-purple-100 mb-8">
              Оптимизируйте работу гидравлических систем с помощью ИИ-аналитики и предикативного
              обслуживания.
            </p>

            <!-- Features -->
            <div class="space-y-4">
              <div class="flex items-center justify-center space-x-3 text-purple-200">
                <Icon name="heroicons:chart-bar-square" class="w-5 h-5" />
                <span class="font-medium">Реальное время мониторинг</span>
              </div>
              <div class="flex items-center justify-center space-x-3 text-purple-200">
                <Icon name="heroicons:cpu-chip" class="w-5 h-5" />
                <span class="font-medium">ИИ-диагностика</span>
              </div>
              <div class="flex items-center justify-center space-x-3 text-purple-200">
                <Icon name="heroicons:wrench-screwdriver" class="w-5 h-5" />
                <span class="font-medium">Предикативное обслуживание</span>
              </div>
              <div class="flex items-center justify-center space-x-3 text-purple-200">
                <Icon name="heroicons:shield-check" class="w-5 h-5" />
                <span class="font-medium">Enterprise безопасность</span>
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
