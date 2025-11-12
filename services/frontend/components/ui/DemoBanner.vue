<!-- Demo mode banner component -->
<template>
  <div 
    v-if="authStore.isDemoMode"
    class="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-purple-600 to-blue-600 text-white py-2 px-4 shadow-lg"
  >
    <div class="container mx-auto flex items-center justify-between">
      <div class="flex items-center gap-3">
        <UIcon name="i-heroicons-play-circle" class="w-5 h-5" />
        <span class="font-medium">
          {{ $t('demo.bannerTitle', 'DEMO MODE') }}
        </span>
        <span class="text-sm opacity-90">
          {{ $t('demo.bannerDesc', 'Explore all features with sample data') }}
        </span>
      </div>
      
      <div class="flex items-center gap-2">
        <UButton
          color="white"
          variant="ghost"
          size="xs"
          @click="exitDemo"
        >
          {{ $t('demo.exitDemo', 'Exit Demo') }}
        </UButton>
        
        <UButton
          color="white"
          variant="solid"
          size="xs"
          @click="goToSignup"
        >
          {{ $t('demo.getStarted', 'Get Started') }}
        </UButton>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
const authStore = useAuthStore()
const router = useRouter()

function exitDemo() {
  authStore.logout()
}

function goToSignup() {
  router.push('/auth/register')
}
</script>
