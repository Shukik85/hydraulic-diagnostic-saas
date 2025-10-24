// API Types for backend integration
export interface User {
  id: number
  username: string
  email: string
  first_name: string
  last_name: string
  company?: string
  position?: string
  phone?: string
  experience_years?: number
  specialization?: string
  email_notifications: boolean
  push_notifications: boolean
  critical_alerts_only: boolean
  created_at: string
  updated_at: string
  last_activity: string
  systems_count: number
  reports_generated: number
  profile?: UserProfile
}

export interface UserProfile {
  user: number
  avatar?: string
  bio?: string
  location?: string
  website?: string
  theme: 'light' | 'dark' | 'auto'
  language: 'ru' | 'en'
  timezone: string
  created_at: string
  updated_at: string
}

export interface HydraulicSystem {
  id: number
  name: string
  description?: string
  system_type: string
  status: 'active' | 'inactive' | 'maintenance'
  location?: string
  installed_date?: string
  owner: number
  components_count: number
  last_reading_at?: string
  metadata: Record<string, any>
  created_at: string
  updated_at: string
}

export interface DiagnosticReport {
  id: number
  system: number
  title: string
  summary: string
  status: 'pending' | 'completed' | 'failed'
  severity: 'low' | 'medium' | 'high' | 'critical'
  recommendations: string
  metadata: Record<string, any>
  generated_by: number
  created_at: string
  updated_at: string
}

export interface SensorData {
  id: number
  system: number
  component: number
  sensor_type: string
  value: number
  unit: string
  timestamp: string
  quality: 'good' | 'suspect' | 'bad'
  metadata: Record<string, any>
}

export interface RagQueryLog {
  id: number
  system: number
  system_name: string
  document?: number
  document_title?: string
  query_text: string
  response_text: string
  timestamp: string
  metadata: Record<string, any>
}

// API Response wrappers
export interface ApiResponse<T> {
  data: T
  message?: string
}

export interface PaginatedResponse<T> {
  count: number
  next?: string
  previous?: string
  results: T[]
}

// Auth types
export interface LoginCredentials {
  email: string
  password: string
}

export interface TokenResponse {
  access: string
  refresh: string
  user: User
}

export interface RegisterData {
  username: string
  email: string
  password: string
  first_name?: string
  last_name?: string
}

// Form validation types
export interface ValidationError {
  field?: string
  message: string
}

export interface ApiError {
  detail: string
  errors?: ValidationError[]
  status_code: number
}