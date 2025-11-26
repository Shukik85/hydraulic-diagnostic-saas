/**
 * Guest middleware
 * Redirects authenticated users to dashboard
 */
export default defineNuxtRouteMiddleware((to, from) => {
  const authStore = useAuthStore();

  if (authStore.isAuthenticated) {
    return navigateTo('/dashboard');
  }
});
