<script setup lang="ts">
import { ref } from 'vue';

import { loginSchema } from '~/utils/validation';

definePageMeta({
  layout: 'auth',
  middleware: 'guest',
});

useSeoMeta({
  title: 'Login - Hydraulic Diagnostic SaaS',
  description: 'Sign in to your account',
});

const authStore = useAuthStore();
const toast = useToast();

const email = ref('');
const password = ref('');
const errors = ref<{ email?: string; password?: string }>({});
const isLoading = ref(false);

const validateForm = (): boolean => {
  errors.value = {};

  const result = loginSchema.safeParse({
    email: email.value,
    password: password.value,
  });

  if (!result.success) {
    result.error.errors.forEach((err) => {
      if (err.path[0]) {
        errors.value[err.path[0] as 'email' | 'password'] = err.message;
      }
    });
    return false;
  }

  return true;
};

const handleSubmit = async (): Promise<void> => {
  if (!validateForm()) {
    return;
  }

  isLoading.value = true;

  try {
    await authStore.login(email.value, password.value);
    toast.success('Login successful', 'Welcome back!');
    await navigateTo('/dashboard');
  } catch (error) {
    toast.error(
      error instanceof Error ? error.message : 'Login failed',
      'Authentication Error'
    );
  } finally {
    isLoading.value = false;
  }
};
</script>

<template>
  <div class="relative flex min-h-screen items-center justify-center overflow-hidden bg-gradient-to-br from-primary-50 via-white to-primary-100 dark:from-gray-900 dark:via-gray-900 dark:to-gray-800">
    <!-- Animated background gradient mesh -->
    <div class="absolute inset-0 overflow-hidden">
      <div class="absolute -left-1/4 -top-1/4 h-96 w-96 animate-pulse rounded-full bg-primary-300/30 blur-3xl dark:bg-primary-600/20" />
      <div class="absolute -bottom-1/4 -right-1/4 h-96 w-96 animate-pulse rounded-full bg-primary-400/30 blur-3xl animation-delay-2000 dark:bg-primary-500/20" />
      <div class="absolute left-1/2 top-1/2 h-96 w-96 -translate-x-1/2 -translate-y-1/2 animate-pulse rounded-full bg-primary-200/20 blur-3xl animation-delay-4000 dark:bg-primary-700/10" />
    </div>

    <!-- Login Card -->
    <div class="relative z-10 w-full max-w-md px-4">
      <div class="rounded-2xl bg-white/80 p-8 shadow-2xl backdrop-blur-xl dark:bg-gray-900/80 dark:shadow-primary-500/10">
        <!-- Logo & Header -->
        <div class="mb-8 text-center">
          <div class="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-primary-500 to-primary-600 shadow-lg">
            <Icon name="heroicons:wrench-screwdriver" class="h-8 w-8 text-white" />
          </div>
          <h1 class="mt-4 text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
            Welcome back
          </h1>
          <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
            Sign in to continue to Hydraulic Diagnostic Platform
          </p>
        </div>

        <!-- Login Form -->
        <form class="space-y-5" @submit.prevent="handleSubmit">
          <Input
            v-model="email"
            type="email"
            label="Email address"
            placeholder="name@company.com"
            :error="errors.email"
            :disabled="isLoading"
            required
            autocomplete="email"
            icon="heroicons:envelope"
          />

          <Input
            v-model="password"
            type="password"
            label="Password"
            placeholder="Enter your password"
            :error="errors.password"
            :disabled="isLoading"
            required
            autocomplete="current-password"
            icon="heroicons:lock-closed"
          />

          <div class="flex items-center justify-between text-sm">
            <label class="flex items-center gap-2 text-gray-700 dark:text-gray-300">
              <input type="checkbox" class="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-2 focus:ring-primary-500 focus:ring-offset-0 dark:border-gray-600 dark:bg-gray-800">
              <span>Remember me</span>
            </label>
            <a href="#" class="font-medium text-primary-600 transition-colors hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300">
              Forgot password?
            </a>
          </div>

          <Button 
            type="submit" 
            variant="primary" 
            size="lg" 
            :loading="isLoading" 
            full-width
            class="mt-6 !bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-700 hover:to-primary-600 shadow-lg shadow-primary-500/30 transition-all hover:shadow-xl hover:shadow-primary-500/40"
          >
            <span class="flex items-center justify-center gap-2">
              <span>Sign in</span>
              <Icon v-if="!isLoading" name="heroicons:arrow-right" class="h-4 w-4" />
            </span>
          </Button>
        </form>

        <!-- Divider -->
        <div class="relative my-8">
          <div class="absolute inset-0 flex items-center">
            <div class="w-full border-t border-gray-300 dark:border-gray-700" />
          </div>
          <div class="relative flex justify-center text-sm">
            <span class="bg-white px-4 text-gray-500 dark:bg-gray-900 dark:text-gray-400">Or</span>
          </div>
        </div>

        <!-- Sign Up Link -->
        <p class="text-center text-sm text-gray-600 dark:text-gray-400">
          Don't have an account?
          <a href="#" class="font-semibold text-primary-600 transition-colors hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300">
            Create account
          </a>
        </p>
      </div>

      <!-- Footer Info -->
      <p class="mt-8 text-center text-xs text-gray-500 dark:text-gray-500">
        Protected by enterprise-grade security
      </p>
    </div>
  </div>
</template>

<style scoped>
@keyframes pulse {
  0%, 100% {
    opacity: 0.6;
  }
  50% {
    opacity: 0.8;
  }
}

.animation-delay-2000 {
  animation-delay: 2s;
}

.animation-delay-4000 {
  animation-delay: 4s;
}
</style>
