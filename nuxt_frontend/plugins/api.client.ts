export default defineNuxtPlugin(() => {
  const authStore = useAuthStore()
  
  $fetch.create({
    onRequest({ request, options }) {
      const headers: Record<string, string> = options.headers as Record<string, string> || {}
      
      if (authStore.isAuthenticated) {
        try {
          // Note: accessing store properties safely
          const token = useCookie('access-token').value
          if (token) {
            headers['Authorization'] = `Bearer ${token}`
          }
        } catch (error) {
          console.warn('Failed to add auth header:', error)
        }
      }
      
      options.headers = headers
    },
    
    async onResponseError({ response }) {
      if (response.status === 401) {
        // Clear auth and redirect to login
        authStore.user = null
        await navigateTo('/auth/login')
      }
      // Return void for proper typing
      return
    }
  })
})