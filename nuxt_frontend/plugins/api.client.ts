export default defineNuxtPlugin(() => {
    const config = useRuntimeConfig()

    const $api = $fetch.create({
        baseURL: config.public.apiBase as string,

        onRequest({ options }) {
            const authStore = useAuthStore()
            if (authStore.accessToken) {
                options.headers = {
                    ...options.headers,
                    Authorization: `Bearer ${authStore.accessToken}`
                }
            }
        },

        onResponseError({ response }) {
            if (response.status === 401) {
                const authStore = useAuthStore()
                authStore.logout()
                if (!String(response.url).includes('/token/')) {
                    return navigateTo('/auth/login')
                }
            }
        }
    })

    return {
        provide: {
            api: $api
        }
    }
})