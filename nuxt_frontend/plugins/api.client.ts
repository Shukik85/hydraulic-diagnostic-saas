export default defineNuxtPlugin(() => {
  const authStore = useAuthStore()
  
  $fetch.create({
    onRequest({ request, options }) {
      // Convert headers to proper type
      const headers: Record<string, string> = {}
      
      // Copy existing headers if any
      if (options.headers) {
        if (options.headers instanceof Headers) {
          options.headers.forEach((value, key) => {
            headers[key] = value
          })
        } else {
          Object.assign(headers, options.headers)
        }
      }
      
      if (authStore.isAuthenticated) {
        try {
          // Access token from cookie safely
          const token = useCookie('access-token').value
          if (token) {
            headers['Authorization'] = `Bearer ${token}`
          }
        } catch (error) {
          console.warn('Failed to add auth header:', error)
        }
      }
      
      // Assign as Record, not Headers class
      options.headers = headers
    },
    
    async onResponseError({ response }) {
      if (response.status === 401) {
        // Clear auth and redirect to login
        authStore.user = null
        await navigateTo('/auth/login')
      }
      // Return void for proper typing
    }
  })
})