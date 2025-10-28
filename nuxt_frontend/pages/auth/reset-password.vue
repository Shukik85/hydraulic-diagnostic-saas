<script setup lang="ts">
// Fixed reset password with proper null safety
definePageMeta({
  layout: 'auth',
  middleware: 'guest',
});

useSeoMeta({
  title: 'Сброс пароля | Hydraulic Diagnostic SaaS',
  robots: 'noindex, nofollow',
});

interface ResetForm {
  password: string;
  confirmPassword: string;
}

const route = useRoute();
const router = useRouter();

// Form state
const form = reactive<ResetForm>({
  password: '',
  confirmPassword: '',
});

const isLoading = ref<boolean>(false);
const error = ref<string>('');
const success = ref<boolean>(false);
const showPassword = ref<boolean>(false);
const showConfirmPassword = ref<boolean>(false);

// Reset token from URL
const token = computed(() => (route.query.token as string) || '');

// Password strength with guaranteed return
const passwordStrength = usePasswordStrength(toRef(form, 'password'));

// Validation
const validation = computed(() => ({
  password: !form.password || form.password.length < 8 ? 'Минимум 8 символов' : '',
  confirmPassword: form.password !== form.confirmPassword ? 'Пароли не совпадают' : '',
}));

const isFormValid = computed(() => {
  return (
    form.password &&
    form.confirmPassword &&
    form.password === form.confirmPassword &&
    form.password.length >= 8 &&
    passwordStrength.value.score >= 3
  );
});

const handlePasswordReset = async () => {
  if (!isFormValid.value || !token.value) return;

  isLoading.value = true;
  error.value = '';

  try {
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));

    success.value = true;
    setTimeout(() => {
      router.push('/auth/login');
    }, 2000);
  } catch (err: any) {
    error.value = err.message || 'Ошибка сброса пароля';
  } finally {
    isLoading.value = false;
  }
};

// Check if we have a valid token
if (!token.value) {
  throw createError({
    statusCode: 400,
    statusMessage: 'Неверная ссылка для сброса пароля',
  });
}
</script>

<template>
  <div class="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8">
      <div class="text-center">
        <div
          class="w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl"
        >
          <Icon name="heroicons:key" class="w-10 h-10 text-white" />
        </div>
        <h1 class="premium-heading-md text-gray-900 dark:text-white mb-2">Новый пароль</h1>
        <p class="premium-body text-gray-600 dark:text-gray-300">Введите новый надёжный пароль</p>
      </div>

      <div v-if="success" class="premium-card p-6 text-center">
        <Icon name="heroicons:check-circle" class="w-16 h-16 text-green-500 mx-auto mb-4" />
        <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">
          Пароль успешно сброшен!
        </h2>
        <p class="text-sm text-gray-600 dark:text-gray-300">Перенаправляем на страницу входа...</p>
      </div>

      <form v-else @submit.prevent="handlePasswordReset" class="premium-card p-8 space-y-6">
        <!-- Error message -->
        <div
          v-if="error"
          class="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg"
        >
          <div class="flex items-center space-x-3">
            <Icon
              name="heroicons:exclamation-triangle"
              class="w-5 h-5 text-red-600 dark:text-red-400"
            />
            <p class="text-sm text-red-700 dark:text-red-300">{{ error }}</p>
          </div>
        </div>

        <!-- New Password -->
        <div>
          <label for="password" class="premium-label">Новый пароль</label>
          <div class="relative">
            <input
              id="password"
              v-model="form.password"
              :type="showPassword ? 'text' : 'password'"
              required
              :disabled="isLoading"
              :class="[validation.password ? 'premium-input-error' : 'premium-input', 'pr-12']"
              placeholder="Введите новый пароль"
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

          <!-- Fixed password strength indicator -->
          <div v-if="form.password" class="mt-2">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs text-gray-500 dark:text-gray-400">Надёжность</span>
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
              placeholder="Повторите пароль"
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

        <!-- Submit button -->
        <PremiumButton
          type="submit"
          full-width
          size="lg"
          gradient
          :loading="isLoading"
          :disabled="!isFormValid"
          icon="heroicons:key"
        >
          Обновить пароль
        </PremiumButton>

        <!-- Back to login -->
        <div class="text-center">
          <NuxtLink
            to="/auth/login"
            class="text-sm font-medium text-blue-600 hover:text-blue-500 dark:text-blue-400 dark:hover:text-blue-300 transition-colors premium-focus"
          >
            Вернуться ко входу
          </NuxtLink>
        </div>
      </form>
    </div>
  </div>
</template>
