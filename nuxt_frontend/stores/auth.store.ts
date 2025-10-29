interface User {
    id: number
    email: string
    first_name: string
    last_name: string
    is_active: boolean
}

interface LoginCredentials {
    email: string
    password: string
}

interface TokenResponse {
    access: string
    refresh: string
}

export const useAuthStore = defineStore('auth', () => {
    const accessToken = ref<string | null>(null)
    const refreshToken = ref<string | null>(null)
    const user = ref<User | null>(null)
    const isLoading = ref(false)
    const error = ref<string | null>(null)

    const isAuthenticated = computed(() => !!accessToken.value)
    const isAdmin = computed(() => user.value?.is_active === true)

    const login = async (credentials: LoginCredentials) => {
        isLoading.value = true
        error.value = null

        try {
            const { $api } = useNuxtApp()
            const response = await $api<TokenResponse>('/token/', {
                method: 'POST',
                body: credentials
            })

            accessToken.value = response.access
            refreshToken.value = response.refresh

            await fetchProfile()
            await navigateTo('/systems')

            return { success: true }
        } catch (err: any) {
            error.value = err.data?.detail || 'Login failed'
            return { success: false, error: error.value }
        } finally {
            isLoading.value = false
        }
    }

    const logout = () => {
        accessToken.value = null
        refreshToken.value = null
        user.value = null
        error.value = null
        return navigateTo('/auth/login')
    }

    const fetchProfile = async () => {
        try {
            const { $api } = useNuxtApp()
            user.value = await $api<User>('/users/me/')
        } catch (err) {
            console.error('Failed to fetch user profile:', err)
        }
    }

    if (process.client) {
        watch(accessToken, (newToken) => {
            if (newToken) {
                localStorage.setItem('access_token', newToken)
            } else {
                localStorage.removeItem('access_token')
            }
        })

        watch(refreshToken, (newToken) => {
            if (newToken) {
                localStorage.setItem('refresh_token', newToken)
            } else {
                localStorage.removeItem('refresh_token')
            }
        })

        const storedAccess = localStorage.getItem('access_token')
        const storedRefresh = localStorage.getItem('refresh_token')

        if (storedAccess) accessToken.value = storedAccess
        if (storedRefresh) refreshToken.value = storedRefresh

        if (storedAccess) {
            fetchProfile()
        }
    }

    return {
        accessToken: readonly(accessToken),
        refreshToken: readonly(refreshToken),
        user: readonly(user),
        isLoading: readonly(isLoading),
        error: readonly(error),
        isAuthenticated,
        isAdmin,
        login,
        logout,
        fetchProfile
    }
})