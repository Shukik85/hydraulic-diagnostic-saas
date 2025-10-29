<template>
  <div class="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8">
      <div class="text-center">
        <h2 class="premium-heading-xl text-gray-900 dark:text-white">
          Sign in to your account
        </h2>
        <p class="mt-2 premium-body text-gray-600 dark:text-gray-400">
          Hydraulic Diagnostic SaaS
        </p>
      </div>
      
      <div class="premium-card p-8 premium-scale-in">
        <form class="space-y-6" @submit.prevent="handleLogin">
          <div>
            <label for="email" class="premium-label">Email address</label>
            <input
              id="email"
              v-model="form.email"
              name="email"
              type="email"
              autocomplete="email"
              required
              class="premium-input"
              :class="{ 'premium-input-error': authStore.error }"
              placeholder="Enter your email"
            />
          </div>
          
          <div>
            <label for="password" class="premium-label">Password</label>
            <input
              id="password"
              v-model="form.password"
              name="password"
              type="password"
              autocomplete="current-password"
              required
              class="premium-input"
              :class="{ 'premium-input-error': authStore.error }"
              placeholder="Enter your password"
            />
          </div>

          <div v-if="authStore.error" class="premium-error-text">
            {{ authStore.error }}
          </div>

          <div>
            <button
              type="submit"
              :disabled="authStore.isLoading"
              class="w-full premium-button-primary premium-button-lg"
            >
              <span v-if="authStore.isLoading" class="loading-spinner w-4 h-4 mr-2"></span>
              <Icon v-else name="heroicons:lock-closed" class="w-4 h-4 mr-2" />
              Sign in
            </button>
          </div>
          
          <div class="text-center">
            <p class="premium-body text-gray-500 dark:text-gray-400">
              Demo credentials: admin@example.com / password
            </p>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: false,
  middleware: 'guest'
})

const authStore = useAuthStore()

const form = ref({
  email: '',
  password: ''
})

const handleLogin = async () => {
  await authStore.login({
    email: form.value.email,
    password: form.value.password
  })
}
</script>

<style scoped>
.loading-spinner {
  @apply inline-block animate-spin rounded-full border-2 border-solid border-current border-r-transparent;
}
</style>