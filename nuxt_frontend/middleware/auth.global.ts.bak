export default defineNuxtRouteMiddleware((to) => {
    const authStore = useAuthStore()
    const publicRoutes = ['/auth/login', '/auth/register', '/']

    if (!authStore.isAuthenticated && !publicRoutes.includes(to.path)) {
        return navigateTo('/auth/login')
    }

    if (authStore.isAuthenticated && to.path === '/auth/login') {
        return navigateTo('/systems')
    }
})