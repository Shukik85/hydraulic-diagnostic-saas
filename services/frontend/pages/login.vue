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
    // auth.login принимает два отдельных параметра: email и password
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
  <div class="flex min-h-screen items-center justify-center bg-gray-50 px-4 dark:bg-gray-900">
    <div class="w-full max-w-md space-y-8">
      <!-- Header -->
      <div class="text-center">
        <h2 class="text-3xl font-bold text-gray-900 dark:text-white">
          Sign in to your account
        </h2>
        <p class="mt-2 text-sm text-gray-600 dark:text-gray-400">
          Hydraulic Diagnostic SaaS Platform
        </p>
      </div>

      <!-- Login Form -->
      <form class="mt-8 space-y-6" @submit.prevent="handleSubmit">
        <div class="space-y-4">
          <Input
            v-model="email"
            type="email"
            label="Email address"
            placeholder="your@email.com"
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
            placeholder="••••••••"
            :error="errors.password"
            :disabled="isLoading"
            required
            autocomplete="current-password"
            icon="heroicons:lock-closed"
          />
        </div>

        <div class="flex items-center justify-between">
          <div class="text-sm">
            <a href="#" class="font-medium text-primary-600 hover:text-primary-500">
              Forgot your password?
            </a>
          </div>
        </div>

        <Button type="submit" variant="primary" size="lg" :loading="isLoading" full-width>
          Sign in
        </Button>

        <p class="text-center text-sm text-gray-600 dark:text-gray-400">
          Don't have an account?
          <a href="#" class="font-medium text-primary-600 hover:text-primary-500">
            Sign up
          </a>
        </p>
      </form>
    </div>
  </div>
</template>
