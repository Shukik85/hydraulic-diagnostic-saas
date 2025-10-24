export interface User {
  id: number
  email: string
  name: string
  firstname?: string
  lastname?: string
  username?: string
  role?: string
  systemscount?: number
  reportsgenerated?: number
  lastactivity?: string
}

export interface HydraulicSystem {
  id: number
  name: string
  status: string
}

export interface DiagnosticReport {
  id: number
  severity: string
}

export interface SensorData { id: number }
export interface RagQueryLog { id: number }
export interface LoginCredentials { email: string; password: string }
export interface TokenResponse { access: string; refresh: string }
export interface RegisterData { email: string; password: string; username?: string; firstname?: string; lastname?: string; company?: string }
export interface PaginatedResponse<T> { results: T[] }
export interface ApiError { detail: string; status: number }
