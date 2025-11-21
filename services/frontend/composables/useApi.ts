import { useRuntimeConfig, useCookie } from '#imports'
import { computed } from 'vue'

export function useApi() {
    const config = useRuntimeConfig()
    const baseURL = config.public.apiBase as string
    const isAuthenticated = computed(() => !!useCookie('access-token').value)

    async function login(credentials: { email: string; password: string }) {
        return await $fetch('/auth/login', {
            baseURL,
            method: 'POST',
            body: credentials,
            credentials: 'include',
        })
    }

    async function register(payload: any) {
        return await $fetch('/auth/register', {
            baseURL,
            method: 'POST',
            body: payload,
            credentials: 'include',
        })
    }

    async function logout() {
        return await $fetch('/auth/logout', {
            baseURL,
            method: 'POST',
            credentials: 'include',
        })
    }

    async function getCurrentUser() {
        if (!isAuthenticated.value) return null
        return await $fetch('/auth/me', {
            baseURL,
            method: 'GET',
            credentials: 'include',
        })
    }

    async function updateUser(profileData: any) {
        return await $fetch('/auth/me', {
            baseURL,
            method: 'PATCH',
            body: profileData,
            credentials: 'include',
        })
    }

    return {
        isAuthenticated,
        login,
        register,
        logout,
        getCurrentUser,
        updateUser,
    }
}
