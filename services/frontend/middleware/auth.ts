/**
 * Authentication middleware
 * Redirects to login if user is not authenticated
 */
export default defineNuxtRouteMiddleware((to, from) => {
  const authStore = useAuthStore();

  if (!authStore.isAuthenticated) {
    return navigateTo('/login');
  }
});
