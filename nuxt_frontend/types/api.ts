// API Types for Hydraulic Diagnostic SaaS

export interface User {
  id: number
  email: string
  username?: string
  name?: string
  first_name?: string
  last_name?: string
  avatar?: string
  is_active: boolean
  is_staff?: boolean
  date_joined: string
  last_login?: string
}

export interface LoginCredentials {
  email: string
  password: string
}

export interface RegisterData {
  email: string
  password: string
  password_confirm: string
  first_name?: string
  last_name?: string
  username?: string
}

export interface AuthTokens {
  access: string
  refresh: string
}

export interface ApiResponse<T = any> {
  data: T
  message?: string
  error?: string
}

export interface HydraulicSystem {
  id: number
  name: string
  type: 'industrial' | 'mobile' | 'marine' | 'construction' | 'mining' | 'agricultural'
  status: 'active' | 'maintenance' | 'inactive'
  description?: string
  location?: string
  pressure: number
  temperature: number
  flow_rate?: number
  vibration?: number
  health_score: number
  last_update: string
  created_at: string
  updated_at: string
}

export interface DiagnosticSession {
  id: number
  system_id: number
  type: 'full' | 'pressure' | 'temperature' | 'vibration' | 'flow'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  started_at: string
  completed_at?: string
  results?: DiagnosticResult[]
  created_by: number
}

export interface DiagnosticResult {
  id: number
  session_id: number
  metric_type: string
  value: number
  threshold_min?: number
  threshold_max?: number
  status: 'normal' | 'warning' | 'critical'
  recommendations?: string[]
  timestamp: string
}

export interface ChatMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  timestamp: string
  sources?: {
    title: string
    url: string
  }[]
}

export interface ChatSession {
  id: number
  title: string
  description: string
  lastMessage: string
  timestamp: string
  messages: ChatMessage[]
}

export interface ApiError {
  message: string
  status: number
  data?: any
}

// Table component types
export interface TableColumn {
  key: string
  label: string
  sortable?: boolean
  width?: string
  align?: 'left' | 'center' | 'right'
}

// Password strength types  
export interface PasswordStrength {
  score: number
  feedback: {
    warning: string
    suggestions: string[]
  }
  crack_times_seconds: {
    online_throttling_100_per_hour: number
    online_no_throttling_10_per_second: number
    offline_slow_hashing_1e4_per_second: number
    offline_fast_hashing_1e10_per_second: number
  }
  crack_times_display: {
    online_throttling_100_per_hour: string
    online_no_throttling_10_per_second: string
    offline_slow_hashing_1e4_per_second: string
    offline_fast_hashing_1e10_per_second: string
  }
}