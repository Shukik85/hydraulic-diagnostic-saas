// Fixed API composable with proper TypeScript types
import type { User, LoginCredentials, RegisterData, TokenResponse } from '~/types/api'

export const useApi = () => {
  const config = useRuntimeConfig()
  const apiBase = config.public.apiBase
  
  // Auth tokens with proper types
  const accessToken = useCookie<string | null>('access_token', {
    default: () => null,
    httpOnly: false,
    secure: process.env.NODE_ENV === 'production'
  })
  
  const refreshToken = useCookie<string | null>('refresh_token', {
    default: () => null,
    httpOnly: false,
    secure: process.env.NODE_ENV === 'production'
  })
  
  const isAuthenticated = computed(() => !!accessToken.value)
  
  // Fixed headers - use .value to get actual computed value
  const $http = $fetch.create({
    baseURL: apiBase,
    // Fix: Don't pass computed ref directly to headers
    onRequest({ request, options }) {
      if (accessToken.value) {
        options.headers = {
          ...options.headers,
          'Authorization': `Bearer ${accessToken.value}`
        }
      }
      options.headers = {
        ...options.headers,
        'Content-Type': 'application/json'
      }
    },
    onResponseError({ response }) {
      if (response.status === 401) {
        // Clear tokens on auth failure
        accessToken.value = null
        refreshToken.value = null
        navigateTo('/auth/login')
      }
    }
  })
  
  // Auth methods with proper return types
  const login = async (credentials: LoginCredentials): Promise<User | null> => {
    const response = await $http<TokenResponse>('/auth/login', {
      method: 'POST',
      body: credentials
    })
    
    accessToken.value = response.access
    refreshToken.value = response.refresh
    
    // Return user or null (not throwing on undefined)
    return response.user || null
  }
  
  const register = async (userData: RegisterData): Promise<User | null> => {
    const response = await $http<TokenResponse>('/auth/register', {
      method: 'POST',
      body: userData
    })
    
    accessToken.value = response.access
    refreshToken.value = response.refresh
    
    return response.user || null
  }
  
  const logout = async (): Promise<void> => {
    try {
      await $http('/auth/logout', {
        method: 'POST'
      })
    } catch {
      // Ignore logout errors
    } finally {
      accessToken.value = null
      refreshToken.value = null
    }
  }
  
  const getCurrentUser = async (): Promise<User | null> => {
    try {
      const user = await $http<User>('/auth/me')
      return user || null
    } catch {
      return null
    }
  }
  
  const updateUser = async (userData: Partial<User>): Promise<User | null> => {
    try {
      const user = await $http<User>('/auth/me', {
        method: 'PATCH',
        body: userData
      })
      return user || null
    } catch {
      return null
    }
  }
  
  return {
    // State
    isAuthenticated,
    
    // Methods
    login,
    register,
    logout,
    getCurrentUser,
    updateUser,
    
    // Raw HTTP client for other operations
    $http
  }
}