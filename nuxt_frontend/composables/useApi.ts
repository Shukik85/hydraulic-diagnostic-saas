interface LoginCredentials {
  email: string
  password: string
}

export const useApi = () => {
  const config = useRuntimeConfig()
  const accessToken = useCookie<string>('access-token', { httpOnly: false })
  const refreshToken = useCookie<string>('refresh-token', { httpOnly: true })
  
  const isAuthenticated = computed(() => !!accessToken.value)
  
  const createHeaders = (): Record<string, string> => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    }
    
    if (accessToken.value) {
      headers['Authorization'] = `Bearer ${accessToken.value}`
    }
    
    return headers
  }
  
  const api = $fetch.create({
    baseURL: config.public.apiBase,
    headers: createHeaders()
  })
  
  const login = async (credentials: LoginCredentials) => {
    const response = await api<any>('/auth/login/', {
      method: 'POST',
      body: credentials
    })
    
    if (response.access && response.refresh) {
      accessToken.value = response.access
      refreshToken.value = response.refresh
    }
    
    return response.user || response
  }
  
  const register = async (userData: any) => {
    return await api<any>('/auth/register/', {
      method: 'POST',
      body: userData
    })
  }
  
  const logout = async () => {
    try {
      await api('/auth/logout/', { method: 'POST' })
    } finally {
      accessToken.value = null
      refreshToken.value = null
    }
  }
  
  const getCurrentUser = async () => {
    return await api<any>('/auth/user/')
  }
  
  const updateUser = async (userData: any) => {
    return await api<any>('/auth/user/', {
      method: 'PATCH',
      body: userData
    })
  }
  
  const refreshAccessToken = async () => {
    if (!refreshToken.value) throw new Error('No refresh token')
    
    const response = await api<any>('/auth/refresh/', {
      method: 'POST',
      body: { refresh: refreshToken.value }
    })
    
    if (response.access) {
      accessToken.value = response.access
    }
    
    return response
  }
  
  return {
    isAuthenticated,
    login,
    register,
    logout,
    getCurrentUser,
    updateUser,
    refreshAccessToken,
    api
  }
}