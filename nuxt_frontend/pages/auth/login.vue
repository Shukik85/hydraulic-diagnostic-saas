<template>
  <div class="min-h-screen grid lg:grid-cols-2">
    <!-- Left Side - Form -->
    <div class="flex items-center justify-center p-8 bg-background animate-fade-in">
      <div class="w-full max-w-md space-y-8">
        <div class="flex flex-col items-center text-center">
          <div class="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary mb-4 animate-scale-in">
            <Icon name="lucide:gauge" class="h-10 w-10 text-primary-foreground" />
          </div>
          <h1 class="text-3xl font-bold mb-2">Welcome Back</h1>
          <p class="text-muted-foreground">
            Sign in to access Hydraulic Diagnostic Platform
          </p>
        </div>

        <form @submit.prevent="handleSubmit" class="space-y-6 animate-slide-up">
          <div class="space-y-2">
            <UiLabel for="email">Email</UiLabel>
            <UiInput
              id="email"
              type="email"
              placeholder="engineer@company.com"
              v-model="email"
              required
              class="transition-all duration-200 focus:ring-2 focus:ring-primary"
            />
          </div>

          <div class="space-y-2">
            <UiLabel for="password">Password</UiLabel>
            <div class="relative">
              <UiInput
                id="password"
                :type="showPassword ? 'text' : 'password'"
                placeholder="Enter your password"
                v-model="password"
                required
                class="pr-10 transition-all duration-200 focus:ring-2 focus:ring-primary"
              />
              <UiButton
                type="button"
                variant="ghost"
                size="icon"
                class="absolute right-0 top-0 h-full px-3 hover:bg-transparent"
                @click="showPassword = !showPassword"
              >
                <Icon
                  :name="showPassword ? 'lucide:eye-off' : 'lucide:eye'"
                  class="h-4 w-4 text-muted-foreground"
                />
              </UiButton>
            </div>
          </div>

          <div class="flex items-center justify-between">
            <div class="flex items-center space-x-2">
              <UiCheckbox id="remember" v-model="rememberMe" />
              <UiLabel for="remember" class="text-sm">Remember me</UiLabel>
            </div>
            <NuxtLink to="/auth/forgot-password" class="text-sm text-primary hover:underline">
              Forgot password?
            </NuxtLink>
          </div>

          <UiButton type="submit" class="w-full" size="lg" :disabled="isLoading">
            <Icon v-if="isLoading" name="lucide:loader-2" class="mr-2 h-4 w-4 animate-spin" />
            Sign In
          </UiButton>

          <div class="text-center">
            <span class="text-sm text-muted-foreground">Don't have an account? </span>
            <NuxtLink to="/auth/register" class="text-sm text-primary hover:underline">
              Sign up
            </NuxtLink>
          </div>
        </form>
      </div>
    </div>

    <!-- Right Side - Illustration -->
    <div class="hidden lg:flex flex-col justify-center p-12 bg-gradient-to-br from-primary/5 to-secondary/5 relative overflow-hidden animate-fade-in">
      <!-- Background Pattern -->
      <div class="absolute inset-0 opacity-10">
        <div class="absolute top-10 left-10 w-32 h-32 bg-primary rounded-full blur-3xl"></div>
        <div class="absolute bottom-10 right-10 w-40 h-40 bg-secondary rounded-full blur-3xl"></div>
        <div class="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-status-success rounded-full blur-3xl"></div>
      </div>

      <div class="relative z-10 flex flex-col justify-center space-y-6 text-white">
        <div class="space-y-6 animate-slide-up">
          <h2 class="text-4xl font-bold">
            Industrial Hydraulic
            <br />
            Diagnostics Platform
          </h2>
          <p class="text-lg text-white/90 max-w-md">
            Real-time monitoring, AI-powered anomaly detection, and intelligent
            diagnostics for industrial hydraulic systems.
          </p>
          <div class="space-y-4 pt-4">
            <div class="flex items-center gap-3">
              <div class="h-12 w-12 rounded-lg bg-white/20 backdrop-blur flex items-center justify-center">
                <Icon name="lucide:activity" class="h-6 w-6" />
              </div>
              <div>
                <p class="font-medium">Real-time Analytics</p>
                <p class="text-sm text-white/70">
                  Monitor all sensors in real-time
                </p>
              </div>
            </div>
            <div class="flex items-center gap-3">
              <div class="h-12 w-12 rounded-lg bg-white/20 backdrop-blur flex items-center justify-center">
                <Icon name="lucide:brain" class="h-6 w-6" />
              </div>
              <div>
                <p class="font-medium">AI Assistant</p>
                <p class="text-sm text-white/70">
                  Intelligent RAG-powered support
                </p>
              </div>
            </div>
            <div class="flex items-center gap-3">
              <div class="h-12 w-12 rounded-lg bg-white/20 backdrop-blur flex items-center justify-center">
                <Icon name="lucide:zap" class="h-6 w-6" />
              </div>
              <div>
                <p class="font-medium">Automated Diagnostics</p>
                <p class="text-sm text-white/70">
                  Detect anomalies before failures
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const email = ref('')
const password = ref('')
const showPassword = ref(false)
const rememberMe = ref(false)
const isLoading = ref(false)

const handleSubmit = async () => {
  isLoading.value = true
  try {
    // Simulate login
    await new Promise(resolve => setTimeout(resolve, 1000))
    console.log('Login attempt:', { email: email.value, password: password.value, rememberMe: rememberMe.value })
    // Navigate to dashboard
    // await navigateTo('/')
    window.location.href = '/'
  } catch (error) {
    console.error('Login error:', error)
  } finally {
    isLoading.value = false
  }
}
</script>
