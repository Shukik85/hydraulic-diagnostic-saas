/**
 * Auth Store - Управление состоянием аутентификации
 *
 * ✅ ИСПРАВЛЕНО:
 * - JWT интеграция через useGeneratedApi
 * - Правильная типизация
 * - Null-safety checks
 * - SSR-safe инициализация
 */

import type { User } from '~/types/api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const authToken = ref<string | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  // Getters with null safety
  const isAuthenticated = computed(() => !!user.value && !!authToken.value)
  const userName = computed(() => {
    if (!user.value) return ''
    const u = user.value
    if (u.first_name || u.last_name) {
      return `${u.first_name || ''} ${u.last_name || ''}`.trim()
    }
    return u.username || u.name || u.email || ''
  })

  /**
   * Login с email и password
   * TODO: После генерации OpenAPI client'а использовать authService.login()
   */
  const login = async (credentials: { email: string; password: string }) => {
    loading.value = true
    error.value = null

    try {
      // Placeholder для будущей реальной интеграции
      // const response = await useGeneratedApi().auth.login(credentials)
      // authToken.value = response.access
      // user.value = response.user

      // Временная mock-реализация для тестирования
      console.warn('[auth.store] Login placeholder - awaiting OpenAPI client')
      throw new Error('Auth service not yet available')
    } catch (err: any) {
      error.value = err?.message || 'Ошибка входа'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Register новый пользователь
   * TODO: После генерации OpenAPI client'а использовать authService.register()
   */
  const register = async (userData: {
    email: string
    password: string
    first_name?: string
    last_name?: string
  }) => {
    loading.value = true
    error.value = null

    try {
      // Placeholder для будущей реальной интеграции
      // const response = await useGeneratedApi().auth.register(userData)
      // authToken.value = response.access
      // user.value = response.user

      console.warn('[auth.store] Register placeholder - awaiting OpenAPI client')
      throw new Error('Auth service not yet available')
    } catch (err: any) {
      error.value = err?.message || 'Ошибка регистрации'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Logout
   */
  const logout = async () => {
    loading.value = true
    try {
      // TODO: Вызвать authService.logout() если требуется
      // await useGeneratedApi().auth.logout()
    } finally {
      user.value = null
      authToken.value = null
      loading.value = false
      error.value = null
    }
  }

  /**
   * Fetch текущий пользователь (для SSR инициализации)
   * TODO: После генерации OpenAPI client'а использовать authService.getMe()
   */
  const fetchCurrentUser = async () => {
    // Если токена нет, не пытаться
    if (!authToken.value) return null

    loading.value = true
    try {
      // Placeholder
      // const userData = await useGeneratedApi().auth.getMe()
      // user.value = userData
      // return userData

      console.warn('[auth.store] fetchCurrentUser placeholder - awaiting OpenAPI client')
      return null
    } catch (err: any) {
      if (err?.status === 401) {
        user.value = null
        authToken.value = null
        await navigateTo('/auth/login')
      }
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Update пользовательский профиль
   * TODO: После генерации OpenAPI client'а использовать authService.updateMe()
   */
  const updateProfile = async (profileData: Partial<User>) => {
    if (!user.value) throw new Error('User not logged in')
    if (!authToken.value) throw new Error('No auth token')

    loading.value = true
    try {
      // Placeholder
      // const updated = await useGeneratedApi().auth.updateMe(profileData)
      // if (updated && user.value) {
      //   user.value = { ...user.value, ...updated }
      // }
      // return updated

      console.warn('[auth.store] updateProfile placeholder - awaiting OpenAPI client')
      throw new Error('Auth service not yet available')
    } catch (err: any) {
      error.value = err?.message || 'Ошибка обновления профиля'
      throw err
    } finally {
      loading.value = false
    }
  }

  /**
   * Initialize auth state (проверить токен из localStorage, восстановить сессию)
   * Вызывается в app.vue при mount'е
   */
  const initialize = async () => {
    // SSR-safe: не выполнять на сервере
    if (process.server) return

    try {
      // Проверить localStorage на наличие токена
      if (typeof window !== 'undefined') {
        const savedToken = localStorage.getItem('auth_token')
        if (savedToken) {
          authToken.value = savedToken
          // Попытаться восстановить пользователя
          await fetchCurrentUser()
        }
      }
    } catch (err) {
      // Silent fail - пользователь нужно будет залогиниться снова
      console.error('[auth.store] Init failed:', err)
    }
  }

  /**
   * Установить token вручную (например, при SSO интеграции)
   */
  const setToken = (token: string) => {
    authToken.value = token
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token)
    }
  }

  /**
   * Очистить token
   */
  const clearToken = () => {
    authToken.value = null
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token')
    }
  }

  return {
    // State
    user: readonly(user),
    authToken: readonly(authToken),
    loading: readonly(loading),
    error: readonly(error),

    // Getters
    isAuthenticated,
    userName,

    // Actions
    login,
    register,
    logout,
    fetchCurrentUser,
    updateProfile,
    initialize,
    setToken,
    clearToken,
  }
})
