/**
 * Auth Store - User authentication and authorization
 * 
 * Features:
 * - User state management
 * - Login/Register/Logout
 * - Profile updates
 * - Token handling via useApi
 * 
 * @example
 * const authStore = useAuthStore()
 * await authStore.login({ email, password })
 */
import { defineStore } from 'pinia'
import type { User, LoginCredentials, RegisterData } from '~/types/api'

export const useAuthStore = defineStore('auth', () => {
  // ==================== STATE ====================
  
  const user = ref<User | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)
  
  const api = useApi()
  
  // ==================== GETTERS ====================
  
  const isAuthenticated = computed(() => !!user.value)
  
  const userName = computed(() => {
    if (!user.value) return ''
    const u = user.value
    
    if (u.first_name || u.last_name) {
      return `${u.first_name || ''} ${u.last_name || ''}`.trim()
    }
    
    return u.username || u.name || u.email || ''
  })
  
  const userInitials = computed(() => {
    if (!user.value) return ''
    const u = user.value
    
    if (u.first_name && u.last_name) {
      return `${u.first_name[0]}${u.last_name[0]}`.toUpperCase()
    }
    
    if (u.name) {
      const parts = u.name.split(' ')
      if (parts.length >= 2) {
        return `${parts[0][0]}${parts[1][0]}`.toUpperCase()
      }
      return u.name.slice(0, 2).toUpperCase()
    }
    
    return u.email.slice(0, 2).toUpperCase()
  })
  
  // ==================== ACTIONS ====================
  
  /**
   * Login user
   */
  async function login(credentials: LoginCredentials) {
    loading.value = true
    error.value = null
    
    try {
      const userData = await api.login(credentials)
      user.value = userData
      return userData
    } catch (err: any) {
      error.value = err?.message || 'Login failed'
      throw err
    } finally {
      loading.value = false
    }
  }
  
  /**
   * Register new user
   */
  async function register(userData: RegisterData) {
    loading.value = true
    error.value = null
    
    try {
      const newUser = await api.register(userData)
      user.value = newUser
      return newUser
    } catch (err: any) {
      error.value = err?.message || 'Registration failed'
      throw err
    } finally {
      loading.value = false
    }
  }
  
  /**
   * Logout user
   */
  async function logout() {
    loading.value = true
    
    try {
      await api.logout()
    } finally {
      user.value = null
      loading.value = false
      error.value = null
    }
  }
  
  /**
   * Fetch current user data
   */
  async function fetchCurrentUser() {
    if (!api.isAuthenticated.value) return null
    
    loading.value = true
    
    try {
      const userData = await api.getCurrentUser()
      user.value = userData
      return userData
    } catch (err: any) {
      if (err.status === 401) {
        user.value = null
        await navigateTo('/auth/login')
      }
      throw err
    } finally {
      loading.value = false
    }
  }
  
  /**
   * Update user profile
   */
  async function updateProfile(profileData: Partial<User>) {
    if (!user.value) {
      throw new Error('User not logged in')
    }
    
    loading.value = true
    
    try {
      const updated = await api.updateUser(profileData)
      if (updated && user.value) {
        user.value = { ...user.value, ...updated }
      }
      return updated
    } catch (err: any) {
      error.value = err?.message || 'Profile update failed'
      throw err
    } finally {
      loading.value = false
    }
  }
  
  /**
   * Initialize auth state (called on app mount)
   */
  async function initialize() {
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
    user,
    loading,
    error,
    
    // Getters
    isAuthenticated,
    userName,
    userInitials,
    
    // Actions
    login,
    register,
    logout,
    fetchCurrentUser,
    updateProfile,
    initialize,
  }
})
