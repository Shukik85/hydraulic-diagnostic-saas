/**
 * Admin middleware
 * Redirects non-admin users to dashboard
 */
export default defineNuxtRouteMiddleware((to, from) => {
  const authStore = useAuthStore();

  if (!authStore.isAuthenticated) {
    return navigateTo('/login');
  }

  if (!authStore.isAdmin) {
    return navigateTo('/dashboard');
  }
});
