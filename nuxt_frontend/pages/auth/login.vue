<template>
  <div class="min-h-screen u-flex-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full space-y-8">
      <div class="text-center">
        <div class="w-16 h-16 mx-auto rounded-xl bg-gradient-to-br from-blue-600 to-blue-400 flex items-center justify-center shadow-lg mb-6">
          <Icon name="heroicons:lock-closed" class="w-8 h-8 text-white" />
        </div>
        <h2 class="u-h2 mb-2">
          Sign in to your account
        </h2>
        <p class="u-body text-gray-600 dark:text-gray-400">
          Access your Hydraulic Diagnostic dashboard
        </p>
      </div>
      
      <div class="u-card p-8">
        <form class="space-y-6" @submit.prevent="handleLogin">
          <div>
            <label for="email" class="u-label">Email address</label>
            <input
              id="email"
              v-model="form.email"
              name="email"
              type="email"
              autocomplete="email"
              required
              class="u-input"
              :class="{ 'u-input-error': authStore.error }"
              placeholder="Enter your email address"
            />
          </div>
          
          <div>
            <label for="password" class="u-label">Password</label>
            <input
              id="password"
              v-model="form.password"
              name="password"
              type="password"
              autocomplete="current-password"
              required
              class="u-input"
              :class="{ 'u-input-error': authStore.error }"
              placeholder="Enter your password"
            />
          </div>

          <div v-if="authStore.error" class="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
            <div class="flex items-center gap-2">
              <Icon name="heroicons:exclamation-circle" class="w-5 h-5 text-red-600 dark:text-red-400" />
              <p class="text-sm text-red-800 dark:text-red-300">{{ authStore.error }}</p>
            </div>
          </div>

          <div>
            <button
              type="submit"
              :disabled="authStore.isLoading"
              class="w-full u-btn u-btn-primary u-btn-lg"
            >
              <span 
                v-if="authStore.isLoading" 
                class="inline-block w-4 h-4 mr-2 rounded-full border-2 border-solid border-current border-r-transparent" 
                style="animation: spin 1s linear infinite;"
              ></span>
              <Icon v-else name="heroicons:arrow-right-on-rectangle" class="w-4 h-4 mr-2" />
              Sign in
            </button>
          </div>
          
          <div class="u-divider"></div>
          
          <div class="text-center">
            <p class="u-body-sm text-gray-500 dark:text-gray-400 mb-4">
              Demo credentials for testing:
            </p>
            <div class="bg-gray-100 dark:bg-gray-800 rounded-lg p-3">
              <p class="u-body-sm font-mono text-gray-700 dark:text-gray-300">
                Email: <span class="font-semibold">admin@example.com</span><br>
                Password: <span class="font-semibold">password</span>
              </p>
            </div>
          </div>
          
          <div class="text-center">
            <p class="u-body-sm text-gray-600 dark:text-gray-400">
              Don't have an account?
              <NuxtLink to="/auth/register" class="font-medium text-blue-600 dark:text-blue-400 hover:text-blue-500 u-transition-fast">
                Sign up here
              </NuxtLink>
            </p>
          </div>
        </form>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
definePageMeta({
  layout: 'blank',
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
@keyframes spin {
  to { transform: rotate(360deg); }
}
</style>