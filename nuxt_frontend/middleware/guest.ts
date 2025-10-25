// Fixed guest middleware with proper type checking
export default defineNuxtRouteMiddleware((to, from) => {
  const authStore = useAuthStore()
  
  // Fixed: use proper isAuthenticated getter
  if (authStore.isAuthenticated) {
    return navigateTo('/dashboard')
  }
})