export default defineNuxtPlugin(() => {
  const authStore = useAuthStore()
  
  $fetch.create({
    onRequest({ request, options }) {
      // Create proper HeadersInit compatible object
      const headers = new Headers()
      
      // Copy existing headers
      if (options.headers) {
        if (options.headers instanceof Headers) {
          options.headers.forEach((value, key) => {
            headers.set(key, value)
          })
        } else if (Array.isArray(options.headers)) {
          options.headers.forEach(([key, value]) => {
            headers.set(key, String(value))
          })
        } else {
          Object.entries(options.headers).forEach(([key, value]) => {
            headers.set(key, String(value))
          })
        }
      }
      
      // Add default Content-Type if not present
      if (!headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json')
      }
      
      // Add Authorization if authenticated
      if (authStore.isAuthenticated) {
        try {
          const token = useCookie('access-token').value
          if (token) {
            headers.set('Authorization', `Bearer ${token}`)
          }
        } catch (error) {
          console.warn('Failed to add auth header:', error)
        }
      }
      
      // Assign Headers object
      options.headers = headers
    },
    
    async onResponseError({ response }) {
      if (response.status === 401) {
        // Clear auth and redirect to login
        authStore.user = null
        await navigateTo('/auth/login')
      }
    }
  })
})