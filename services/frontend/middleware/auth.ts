/**
 * Auth Middleware
 * Protects routes requiring authentication
 * 
 * DEV MODE: Set NUXT_PUBLIC_DEV_SKIP_AUTH=true in .env to bypass auth in development
 */

export default defineNuxtRouteMiddleware((to, from) => {
  const authStore = useAuthStore();
  const config = useRuntimeConfig();

  // DEV MODE: Skip auth check if explicitly enabled
  const skipAuth = config.public.devSkipAuth === 'true' || config.public.devSkipAuth === true;
  
  if (skipAuth && import.meta.dev) {
    console.warn('[DEV MODE] Auth middleware bypassed - NUXT_PUBLIC_DEV_SKIP_AUTH is enabled');
    return;
  }

  // Check if user is authenticated
  if (!authStore.isAuthenticated) {
    // Redirect to login with return URL
    return navigateTo({
      path: '/login',
      query: { redirect: to.fullPath },
    });
  }
});
