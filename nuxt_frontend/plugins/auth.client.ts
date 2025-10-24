// Auth initialization plugin
export default defineNuxtPlugin(async () => {
  const authStore = useAuthStore()
  
  // Initialize auth state from stored tokens
  if (process.client) {
    await authStore.initialize()
  }
})