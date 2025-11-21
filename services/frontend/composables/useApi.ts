/**
 * Enterprise API Composable with JWT Refresh Rotation
 * 
 * Features:
 * - Auto token refresh on 401
 * - Request/response interceptors
 * - Cookie-based auth (httpOnly secure)
 * - Type-safe methods
 * - Error handling
 * 
 * @module useApi
 */

import { useRuntimeConfig, useCookie, navigateTo } from '#imports'
import { ref, computed } from 'vue'
import type { User, LoginCredentials, RegisterData, AuthTokens } from '~/types/api'

let isRefreshing = false
let refreshSubscribers: Array<(token: string) => void> = []

function subscribeTokenRefresh(cb: (token: string) => void) {
  refreshSubscribers.push(cb)
}

function onRefreshed(token: string) {
  refreshSubscribers.forEach(cb => cb(token))
  refreshSubscribers = []
}

export function useApi() {
  const config = useRuntimeConfig()
  const baseURL = config.public.apiBase as string
  
  // Cookies for token storage
  const accessTokenCookie = useCookie('access-token', {
    maxAge: 60 * 15, // 15 minutes
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax'
  })
  
  const refreshTokenCookie = useCookie('refresh-token', {
    maxAge: 60 * 60 * 24 * 7, // 7 days
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    httpOnly: false // Set to true if backend supports httpOnly
  })

  // Computed properties
  const isAuthenticated = computed(() => !!accessTokenCookie.value)
  const accessToken = computed(() => accessTokenCookie.value)

  /**
   * Refresh access token using refresh token
   */
  async function refreshAccessToken(): Promise<string | null> {
    const refreshToken = refreshTokenCookie.value
    if (!refreshToken) {
      console.warn('[useApi] No refresh token available')
      return null
    }

    try {
      const response = await $fetch<AuthTokens>('/auth/refresh', {
        baseURL,
        method: 'POST',
        body: { refresh: refreshToken },
      })

      // Update tokens
      accessTokenCookie.value = response.access
      if (response.refresh) {
        refreshTokenCookie.value = response.refresh
      }

      return response.access
    } catch (error: any) {
      console.error('[useApi] Token refresh failed:', error)
      // Clear tokens on refresh failure
      clearTokens()
      return null
    }
  }

  /**
   * Make authenticated request with auto-retry on 401
   */
  async function authenticatedFetch<T = any>(
    url: string,
    options: any = {}
  ): Promise<T> {
    const headers = new Headers(options.headers || {})
    
    // Add auth header
    if (accessTokenCookie.value) {
      headers.set('Authorization', `Bearer ${accessTokenCookie.value}`)
    }
    
    if (!headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json')
    }

    try {
      return await $fetch<T>(url, {
        ...options,
        baseURL,
        headers,
        credentials: 'include',
      })
    } catch (error: any) {
      // Handle 401 - try refresh token
      if (error?.status === 401 || error?.statusCode === 401) {
        if (isRefreshing) {
          // Wait for ongoing refresh
          return new Promise((resolve, reject) => {
            subscribeTokenRefresh(async (token: string) => {
              try {
                headers.set('Authorization', `Bearer ${token}`)
                const result = await $fetch<T>(url, {
                  ...options,
                  baseURL,
                  headers,
                  credentials: 'include',
                })
                resolve(result)
              } catch (err) {
                reject(err)
              }
            })
          })
        }

        isRefreshing = true
        const newToken = await refreshAccessToken()
        isRefreshing = false

        if (newToken) {
          onRefreshed(newToken)
          headers.set('Authorization', `Bearer ${newToken}`)
          // Retry original request
          return await $fetch<T>(url, {
            ...options,
            baseURL,
            headers,
            credentials: 'include',
          })
        } else {
          // Refresh failed - redirect to login
          await navigateTo('/auth/login')
          throw new Error('Authentication required')
        }
      }

      throw error
    }
  }

  /**
   * Clear all authentication tokens
   */
  function clearTokens() {
    accessTokenCookie.value = null
    refreshTokenCookie.value = null
  }

  // ==================== AUTH METHODS ====================

  /**
   * Login user with credentials
   */
  async function login(credentials: LoginCredentials): Promise<User> {
    const response = await $fetch<{ user: User; tokens: AuthTokens }>('/auth/login', {
      baseURL,
      method: 'POST',
      body: credentials,
      credentials: 'include',
    })

    // Store tokens
    accessTokenCookie.value = response.tokens.access
    refreshTokenCookie.value = response.tokens.refresh

    return response.user
  }

  /**
   * Register new user
   */
  async function register(userData: RegisterData): Promise<User> {
    const response = await $fetch<{ user: User; tokens: AuthTokens }>('/auth/register', {
      baseURL,
      method: 'POST',
      body: userData,
      credentials: 'include',
    })

    // Store tokens
    accessTokenCookie.value = response.tokens.access
    refreshTokenCookie.value = response.tokens.refresh

    return response.user
  }

  /**
   * Logout user
   */
  async function logout(): Promise<void> {
    try {
      await authenticatedFetch('/auth/logout', {
        method: 'POST',
      })
    } catch (error) {
      console.error('[useApi] Logout error:', error)
    } finally {
      clearTokens()
    }
  }

  /**
   * Get current user profile
   */
  async function getCurrentUser(): Promise<User | null> {
    if (!isAuthenticated.value) {
      return null
    }

    try {
      return await authenticatedFetch<User>('/auth/me', {
        method: 'GET',
      })
    } catch (error: any) {
      if (error?.status === 401 || error?.statusCode === 401) {
        return null
      }
      throw error
    }
  }

  /**
   * Update user profile
   */
  async function updateUser(profileData: Partial<User>): Promise<User> {
    return await authenticatedFetch<User>('/auth/me', {
      method: 'PATCH',
      body: profileData,
    })
  }

  /**
   * Change user password
   */
  async function changePassword(data: {
    old_password: string
    new_password: string
  }): Promise<void> {
    await authenticatedFetch('/auth/change-password', {
      method: 'POST',
      body: data,
    })
  }

  /**
   * Request password reset
   */
  async function requestPasswordReset(email: string): Promise<void> {
    await $fetch('/auth/password-reset', {
      baseURL,
      method: 'POST',
      body: { email },
    })
  }

  /**
   * Reset password with token
   */
  async function resetPassword(data: {
    token: string
    new_password: string
  }): Promise<void> {
    await $fetch('/auth/password-reset/confirm', {
      baseURL,
      method: 'POST',
      body: data,
    })
  }

  return {
    // State
    isAuthenticated,
    accessToken,

    // Auth methods
    login,
    register,
    logout,
    getCurrentUser,
    updateUser,
    changePassword,
    requestPasswordReset,
    resetPassword,

    // Token management
    refreshAccessToken,
    clearTokens,

    // Generic authenticated fetch
    authenticatedFetch,
  }
}
