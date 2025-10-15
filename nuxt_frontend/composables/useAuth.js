// composables/useAuth.js
import { ref, computed } from 'vue'

/**
 * Базовый composable для аутентификации пользователей
 * Содержит состояние пользователя, токена и методы login/logout/refresh
 * Для продакшена: подключите реальный API и secure-хранение токена (cookies/httpOnly)
 */
export const useAuth = () => {
  const user = ref(null)
  const token = ref(null)
  const loading = ref(false)
  const error = ref(null)

  const isAuthenticated = computed(() => Boolean(token.value))

  const config = useRuntimeConfig()
  const API_BASE_URL = config.public.apiBaseUrl || 'http://localhost:8000/api'

  // Пример логина (замените на реальный эндпоинт)
  const login = async ({ email, password }) => {
    loading.value = true
    error.value = null
    try {
      // Пример запроса — настройте под ваш бэкенд
      const resp = await $fetch(`${API_BASE_URL}/auth/login/`, {
        method: 'POST',
        body: { email, password },
        headers: { 'Content-Type': 'application/json' },
      })
      token.value = resp?.token || resp?.access || null
      user.value = resp?.user || null
      return resp
    } catch (err) {
      console.error('Login error:', err)
      error.value = err?.data?.detail || err.message || 'Не удалось выполнить вход'
      token.value = null
      user.value = null
      throw err
    } finally {
      loading.value = false
    }
  }

  const logout = async () => {
    loading.value = true
    error.value = null
    try {
      // Если нужен запрос на сервер для инвалидирования токена — добавьте его тут
      token.value = null
      user.value = null
    } catch (err) {
      console.error('Logout error:', err)
      error.value = err.message || 'Ошибка при выходе'
    } finally {
      loading.value = false
    }
  }

  const refreshUser = async () => {
    if (!token.value) return null
    loading.value = true
    error.value = null
    try {
      const resp = await $fetch(`${API_BASE_URL}/auth/me/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token.value}`,
        },
      })
      user.value = resp || null
      return resp
    } catch (err) {
      console.error('Refresh user error:', err)
      error.value = err.message || 'Не удалось обновить пользователя'
      return null
    } finally {
      loading.value = false
    }
  }

  return {
    // state
    user,
    token,
    loading,
    error,
    isAuthenticated,
    // actions
    login,
    logout,
    refreshUser,
  }
}
