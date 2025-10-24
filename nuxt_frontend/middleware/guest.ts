// Guest middleware - redirect authenticated users away from auth pages
export default defineNuxtRouteMiddleware(() => {
  const authStore = useAuthStore()
  
  // If user is already authenticated, redirect to dashboard
  if (authStore.isAuthenticated) {
    return navigateTo('/dashboard')
  }
})