// Authentication middleware with dev stub
export default defineNuxtRouteMiddleware(to => {
  // DEV MODE: Skip auth check for easier testing
  if (process.dev) {
    console.log('🔓 Auth middleware: DEV mode - skipping auth check for', to.path)
    return
  }
  
  // PRODUCTION: Proper auth check
  const { $router } = useNuxtApp()
  const api = useApi()

  // Check if user is authenticated
  if (!api.isAuthenticated.value) {
    // Redirect to login page, preserving the intended destination
    const redirectTo = to.fullPath !== '/auth/login' ? to.fullPath : '/'
    return navigateTo(`/auth/login?redirect=${encodeURIComponent(redirectTo)}`)
  }
})