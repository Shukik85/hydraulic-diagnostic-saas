// Authentication middleware with demo and dev modes
export default defineNuxtRouteMiddleware(to => {
  const config = useRuntimeConfig()
  
  // ✅ Public routes - always accessible
  const publicRoutes = [
    '/auth/login',
    '/auth/register',
    '/',
    '/demo',
    '/features',
    '/pricing',
  ]
  
  if (publicRoutes.includes(to.path)) {
    return
  }

  // ✅ DEMO MODE: Auto-login with demo user
  if (config.public.demoMode) {
    console.log('🎭 DEMO mode: Auto-authenticated for', to.path)
    
    // Auto-initialize demo user if needed
    const authStore = useAuthStore()
    if (!authStore.isAuthenticated) {
      authStore.loginAsDemo()
    }
    
    return
  }

  // ✅ DEV MODE: Auto-bypass for development
  if (process.dev) {
    console.log('🔓 DEV mode: Auto-authenticated for', to.path)
    return
  }

  // ✅ PRODUCTION: Real auth check
  try {
    const authStore = useAuthStore()
    
    if (!authStore.isAuthenticated) {
      const redirectTo = to.fullPath !== '/auth/login' ? to.fullPath : '/'
      return navigateTo(`/auth/login?redirect=${encodeURIComponent(redirectTo)}`)
    }
  } catch (error) {
    console.error('Auth middleware error:', error)
    if (process.dev) return
    return navigateTo('/auth/login')
  }
})
