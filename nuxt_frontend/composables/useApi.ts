// API client composable for backend integration
import type { 
  User, 
  HydraulicSystem, 
  DiagnosticReport, 
  SensorData,
  RagQueryLog,
  LoginCredentials,
  TokenResponse,
  RegisterData,
  PaginatedResponse,
  ApiError
} from '~/types/api'

export const useApi = () => {
  const config = useRuntimeConfig()
  const apiBase = config.public.apiBase
  
  // Auth token management
  const accessToken = useCookie('access_token', { secure: true, sameSite: 'strict' })
  const refreshToken = useCookie('refresh_token', { secure: true, sameSite: 'strict' })
  
  const isAuthenticated = computed(() => !!accessToken.value)
  
  // HTTP client with auth headers
  const $http = $fetch.create({
    baseURL: apiBase,
    headers: computed(() => ({
      'Content-Type': 'application/json',
      ...(accessToken.value && { Authorization: `Bearer ${accessToken.value}` })
    })),
    onResponseError({ response }) {
      if (response.status === 401) {
        // Token expired, clear auth and redirect to login
        accessToken.value = null
        refreshToken.value = null
        navigateTo('/auth/login')
      }
    }
  })
  
  // Auth methods
  const login = async (credentials: LoginCredentials): Promise<User> => {
    const response = await $http<TokenResponse>('/auth/login/', {
      method: 'POST',
      body: credentials
    })
    
    accessToken.value = response.access
    refreshToken.value = response.refresh
    
    return response.user
  }
  
  const register = async (userData: RegisterData): Promise<User> => {
    const response = await $http<TokenResponse>('/auth/register/', {
      method: 'POST',
      body: userData
    })
    
    accessToken.value = response.access
    refreshToken.value = response.refresh
    
    return response.user
  }
  
  const logout = async () => {
    try {
      await $http('/auth/logout/', { method: 'POST' })
    } finally {
      accessToken.value = null
      refreshToken.value = null
      await navigateTo('/auth/login')
    }
  }
  
  // User methods
  const getCurrentUser = () => $http<User>('/users/me/')
  const updateUser = (data: Partial<User>) => $http<User>('/users/me/', {
    method: 'PATCH',
    body: data
  })
  const changePassword = (oldPassword: string, newPassword: string) => 
    $http('/users/me/change_password/', {
      method: 'POST',
      body: { old_password: oldPassword, new_password: newPassword }
    })
  
  // Hydraulic systems
  const getSystems = (params?: Record<string, any>) => 
    $http<PaginatedResponse<HydraulicSystem>>('/systems/', { params })
  const getSystem = (id: number) => $http<HydraulicSystem>(`/systems/${id}/`)
  const createSystem = (data: Omit<HydraulicSystem, 'id' | 'created_at' | 'updated_at' | 'owner' | 'components_count' | 'last_reading_at'>) => 
    $http<HydraulicSystem>('/systems/', { method: 'POST', body: data })
  const updateSystem = (id: number, data: Partial<HydraulicSystem>) => 
    $http<HydraulicSystem>(`/systems/${id}/`, { method: 'PATCH', body: data })
  const deleteSystem = (id: number) => $http(`/systems/${id}/`, { method: 'DELETE' })
  
  // Diagnostic reports
  const getReports = (systemId?: number, params?: Record<string, any>) => {
    const url = systemId ? `/systems/${systemId}/reports/` : '/reports/'
    return $http<PaginatedResponse<DiagnosticReport>>(url, { params })
  }
  const getReport = (id: number) => $http<DiagnosticReport>(`/reports/${id}/`)
  const createReport = (systemId: number) => 
    $http<DiagnosticReport>(`/systems/${systemId}/reports/`, { method: 'POST' })
  
  // Sensor data
  const getSensorData = (systemId: number, params?: Record<string, any>) => 
    $http<PaginatedResponse<SensorData>>(`/systems/${systemId}/sensor-data/`, { params })
  const addSensorData = (systemId: number, data: Omit<SensorData, 'id' | 'system'>) => 
    $http<SensorData>(`/systems/${systemId}/sensor-data/`, { method: 'POST', body: data })
  
  // RAG Assistant
  const queryRag = (systemId: number, query: string) => 
    $http<{ response: string, sources?: any[] }>(`/systems/${systemId}/rag/query/`, {
      method: 'POST',
      body: { query }
    })
  const getRagLogs = (systemId?: number, params?: Record<string, any>) => {
    const url = systemId ? `/systems/${systemId}/rag/logs/` : '/rag/logs/'
    return $http<PaginatedResponse<RagQueryLog>>(url, { params })
  }
  
  return {
    // Auth
    login,
    register,
    logout,
    isAuthenticated,
    
    // Users
    getCurrentUser,
    updateUser,
    changePassword,
    
    // Systems
    getSystems,
    getSystem,
    createSystem,
    updateSystem,
    deleteSystem,
    
    // Reports
    getReports,
    getReport,
    createReport,
    
    // Sensor data
    getSensorData,
    addSensorData,
    
    // RAG
    queryRag,
    getRagLogs
  }
}