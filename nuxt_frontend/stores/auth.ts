// Authentication state management with Pinia
import type { User } from '~/types/api'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  
  const api = useApi()
  
  // Getters
  const isLoggedIn = computed(() => !!user.value)
  const userName = computed(() => {
    if (!user.value) return ''
    return user.value.first_name || user.value.last_name 
      ? `${user.value.first_name} ${user.value.last_name}`.trim()
      : user.value.username
  })
  
  // Actions
  const login = async (email: string, password: string) => {
    loading.value = true
    error.value = null
    
    try {
      const userData = await api.login({ email, password })
      user.value = userData
      return userData
    } catch (err: any) {
      error.value = err?.data?.detail || 'Ошибка входа'
      throw err
    } finally {
      loading.value = false
    }
  }
  
  const register = async (userData: { username: string, email: string, password: string, first_name?: string, last_name?: string }) => {
    loading.value = true
    error.value = null
    
    try {
      const newUser = await api.register(userData)
      user.value = newUser
      return newUser
    } catch (err: any) {
      error.value = err?.data?.detail || 'Ошибка регистрации'
      throw err
    } finally {
      loading.value = false
    }
  }
  
  const logout = async () => {
    loading.value = true
    try {
      await api.logout()
    } finally {
      user.value = null
      loading.value = false
      error.value = null
    }
  }
  
  const fetchCurrentUser = async () => {
    if (!api.isAuthenticated.value) return null
    
    loading.value = true
    try {
      const userData = await api.getCurrentUser()
      user.value = userData
      return userData
    } catch (err: any) {
      if (err.status === 401) {
        // Token expired
        user.value = null
        await navigateTo('/auth/login')
      }
      throw err
    } finally {
      loading.value = false
    }
  }
  
  const updateProfile = async (profileData: Partial<User>) => {
    if (!user.value) throw new Error('User not logged in')
    
    loading.value = true
    try {
      const updated = await api.updateUser(profileData)
      user.value = { ...user.value, ...updated }
      return updated
    } catch (err: any) {
      error.value = err?.data?.detail || 'Ошибка обновления профиля'
      throw err
    } finally {
      loading.value = false
    }
  }
  
  // Initialize auth state on app load
  const initialize = async () => {
    if (process.server) return
    
    if (api.isAuthenticated.value) {
      try {
        await fetchCurrentUser()
      } catch {
        // Silent fail - user will need to login again
      }
    }
  }
  
  return {
    // State
    user: readonly(user),
    loading: readonly(loading),
    error: readonly(error),
    
    // Getters  
    isLoggedIn,
    userName,
    
    // Actions
    login,
    register,
    logout,
    fetchCurrentUser,
    updateProfile,
    initialize
  }
})