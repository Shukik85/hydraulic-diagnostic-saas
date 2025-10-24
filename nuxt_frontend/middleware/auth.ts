// Authentication middleware
export default defineNuxtRouteMiddleware((to) => {
  const { $router } = useNuxtApp()
  const api = useApi()
  
  // Check if user is authenticated
  if (!api.isAuthenticated.value) {
    // Redirect to login page, preserving the intended destination
    const redirectTo = to.fullPath !== '/auth/login' ? to.fullPath : '/'
    return navigateTo(`/auth/login?redirect=${encodeURIComponent(redirectTo)}`)
  }
})