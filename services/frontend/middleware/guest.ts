/**
 * Guest Middleware
 * Redirects authenticated users away from guest pages (login, register)
 * 
 * DEV MODE: Set NUXT_PUBLIC_DEV_SKIP_AUTH=true in .env to bypass in development
 */

export default defineNuxtRouteMiddleware((to, from) => {
  const authStore = useAuthStore();
  const config = useRuntimeConfig();

  // DEV MODE: Skip guest check if explicitly enabled
  const skipAuth = config.public.devSkipAuth === 'true' || config.public.devSkipAuth === true;
  
  if (skipAuth && import.meta.dev) {
    console.warn('[DEV MODE] Guest middleware bypassed - NUXT_PUBLIC_DEV_SKIP_AUTH is enabled');
    return;
  }

  // Redirect authenticated users to dashboard
  if (authStore.isAuthenticated) {
    return navigateTo('/dashboard');
  }
});
