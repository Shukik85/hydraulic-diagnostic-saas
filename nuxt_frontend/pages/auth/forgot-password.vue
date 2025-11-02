<script setup lang="ts">
// Password recovery page
definePageMeta({
  layout: 'auth',
  middleware: 'guest',
});

useSeoMeta({
  title: 'Восстановление пароля | Hydraulic Diagnostic SaaS',
  description:
    'Restore access to your hydraulic diagnostics platform account with secure password recovery.',
  robots: 'noindex, nofollow',
});

const form = reactive({
  email: '',
});

const isLoading = ref(false);
const isEmailSent = ref(false);
const error = ref('');

const isFormValid = computed(() => {
  return form.email && form.email.includes('@');
});

const handlePasswordReset = async () => {
  if (!isFormValid.value) {
    error.value = 'Введите корректный email адрес';
    return;
  }

  isLoading.value = true;
  error.value = '';

  try {
    // Call password reset API
    // await authStore.resetPassword(form.email)

    // For demo - simulate success
    await new Promise(resolve => setTimeout(resolve, 2000));
    isEmailSent.value = true;
  } catch (err: any) {
    console.error('Password reset error:', err);
    error.value = err.message || 'Ошибка отправки. Попробуйте позже.';
  } finally {
    isLoading.value = false;
  }
};

const emailInput = ref<HTMLInputElement>();

onMounted(() => {
  emailInput.value?.focus();
});
</script>

<template>
  <div class="min-h-screen flex items-center justify-center">
    <div class="max-w-md w-full space-y-8 px-4">
      <!-- Logo and title -->
      <div class="text-center">
        <div
          class="w-20 h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-xl"
        >
          <Icon name="heroicons:key" class="w-10 h-10 text-white" />
        </div>

        <h1 class="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Восстановление доступа
        </h1>
        <p class="text-gray-600 dark:text-gray-300">
          Введите email, чтобы получить ссылку для смены пароля
        </p>
      </div>

      <!-- Success state -->
      <div v-if="isEmailSent" class="text-center space-y-6">
        <div
          class="w-16 h-16 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center mx-auto"
        >
          <Icon name="heroicons:check-circle" class="w-8 h-8 text-green-600 dark:text-green-400" />
        </div>

        <div>
          <h2 class="text-xl font-bold text-gray-900 dark:text-white mb-2">Письмо отправлено!</h2>
          <p class="text-gray-600 dark:text-gray-300 mb-4">
            Мы отправили инструкции по смене пароля на адрес:
          </p>
          <p class="font-medium text-blue-600 dark:text-blue-400">{{ form.email }}</p>
          <p class="text-sm text-gray-500 dark:text-gray-400 mt-4">
            Не получили письмо? Проверьте папку спам или попробуйте ещё раз.
          </p>
        </div>

        <div class="space-y-3">
          <button
            @click="
              isEmailSent = false;
              form.email = '';
            "
            class="w-full py-3 px-4 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 font-medium rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Отправить повторно
          </button>

          <NuxtLink
            to="/auth/login"
            class="w-full inline-flex justify-center py-3 px-4 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg transition-colors"
          >
            Вернуться ко входу
          </NuxtLink>
        </div>
      </div>

      <!-- Reset form -->
      <form v-else @submit.prevent="handlePasswordReset" class="space-y-6">
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

        <!-- Email field -->
        <div>
          <label for="email" class="block text-sm font-semibold text-gray-900 dark:text-white mb-2">
            Электронная почта
          </label>
          <div class="relative">
            <input
              id="email"
              ref="emailInput"
              v-model="form.email"
              type="email"
              autocomplete="email"
              required
              :disabled="isLoading"
              class="w-full px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors disabled:opacity-50 bg-white dark:bg-gray-800 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              placeholder="your.email@company.com"
            />
            <Icon
              name="heroicons:at-symbol"
              class="absolute right-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400"
            />
          </div>
          <p class="mt-2 text-xs text-gray-500 dark:text-gray-400">
            Мы отправим вам ссылку для создания нового пароля
          </p>
        </div>

        <!-- Submit button -->
        <button
          type="submit"
          :disabled="!isFormValid || isLoading"
          class="w-full flex justify-center py-3 px-4 border border-transparent text-base font-bold rounded-lg text-white bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 hover:scale-105"
        >
          <span v-if="!isLoading" class="flex items-center">
            <Icon name="heroicons:envelope" class="w-5 h-5 mr-2" />
            Отправить ссылку
          </span>

          <span v-else class="flex items-center">
            <Icon name="heroicons:arrow-path" class="w-5 h-5 mr-2 animate-spin" />
            Отправляем...
          </span>
        </button>
      </form>

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

      <!-- Support contact -->
      <div
        class="mt-8 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg"
      >
        <div class="text-center">
          <p class="text-sm text-blue-900 dark:text-blue-100 mb-2">Нужна помощь?</p>
          <div class="flex items-center justify-center space-x-4">
            <a
              href="tel:+74959847621"
              class="flex items-center space-x-2 text-blue-700 dark:text-blue-300 hover:text-blue-800 dark:hover:text-blue-200 transition-colors"
            >
              <Icon name="heroicons:phone" class="w-4 h-4" />
              <span class="text-sm font-medium">+7 (495) 984-76-21</span>
            </a>
            <span class="text-blue-600 dark:text-blue-400">•</span>
            <a
              href="mailto:support@hydraulic-diagnostics.com"
              class="flex items-center space-x-2 text-blue-700 dark:text-blue-300 hover:text-blue-800 dark:hover:text-blue-200 transition-colors"
            >
              <Icon name="heroicons:envelope" class="w-4 h-4" />
              <span class="text-sm font-medium">Поддержка</span>
            </a>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
