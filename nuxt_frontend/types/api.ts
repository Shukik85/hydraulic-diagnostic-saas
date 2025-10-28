// Enhanced API types with proper nullable handling

// User and Authentication Types
export interface User {
  id: number;
  email: string;
  name: string;
  first_name?: string;
  last_name?: string;
  username?: string;
  role?: 'admin' | 'operator' | 'viewer' | 'investor';
  is_active?: boolean;
  date_joined?: string;
  last_login?: string;

  // Business metrics
  systems_count?: number;
  reports_generated?: number;
  last_activity?: string;

  // Additional fields for complete coverage
  company?: string;
  phone?: string;
  job_title?: string;
}

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface RegisterData {
  email: string;
  password: string;
  password_confirm?: string;
  first_name?: string;
  last_name?: string;
  username?: string;
  company?: string;
  phone?: string;
  job_title?: string;
  subscribe_updates?: boolean;
  terms_accepted?: boolean;
  newsletter_subscription?: boolean;
}

export interface TokenResponse {
  access: string;
  refresh: string;
  user?: User; // Optional because might not be included
}

export interface PasswordResetRequest {
  email: string;
}

export interface PasswordResetConfirm {
  token: string;
  password: string;
  password_confirm: string;
}

// Hydraulic System Types
export interface HydraulicSystem {
  id: number;
  name: string;
  system_type: string;
  status: 'active' | 'maintenance' | 'inactive' | 'warning' | 'critical';
  location?: string;
  installation_date?: string;
  last_maintenance?: string;
  next_maintenance?: string;
  components_count?: number;
  sensors_count?: number;
  alerts_count?: number;
  uptime_percentage?: number;
  efficiency_score?: number;
  last_reading_at?: string; // Added field

  // Operational data
  temperature?: number;
  pressure?: number;
  flow_rate?: number;
  vibration_level?: number;

  // Metadata
  created_at?: string;
  updated_at?: string;
  owner?: User;
}

// Diagnostic and Sensor Types
export interface DiagnosticReport {
  id: number;
  system_id: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  title: string;
  description?: string;
  summary?: string; // Added field
  recommendations?: string[];
  estimated_cost?: number;
  priority_score?: number;

  // Timestamps
  created_at: string;
  updated_at?: string;
  completed_at?: string;

  // Relations
  system?: HydraulicSystem;
  generated_by?: User;
}

export interface SensorData {
  id: number;
  sensor_id: string;
  system_id: number;
  value: number;
  unit: string;
  sensor_type: 'temperature' | 'pressure' | 'flow' | 'vibration' | 'ph' | 'conductivity';
  threshold_min?: number;
  threshold_max?: number;
  status: 'normal' | 'warning' | 'critical';

  // Metadata
  timestamp: string;
  location?: string;
  calibration_date?: string;

  // Relations
  system?: HydraulicSystem;
}

// RAG Assistant Types
export interface RagQueryLog {
  id: number;
  query: string;
  response?: string;
  context_sources?: string[];
  relevance_score?: number;
  response_time_ms?: number;
  status: 'pending' | 'completed' | 'failed';

  // User context
  user_id?: number;
  session_id?: string;

  // Timestamps
  created_at: string;
  completed_at?: string;

  // Relations
  user?: User;
}

// Chat Types
export interface ChatMessage {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: { title: string; url: string }[];
}

export interface ChatSession {
  id: number;
  title: string;
  description: string;
  lastMessage: string;
  timestamp: string;
  messages: ChatMessage[];
}

// Password Strength
export interface PasswordStrength {
  score: number;
  label: string;
  color: 'red' | 'yellow' | 'green' | 'gray';
}

// UI Types
export interface TableColumn {
  key: string;
  label: string;
  sortable?: boolean;
}

export type ButtonColor = 'blue' | 'green' | 'purple' | 'orange' | 'teal' | 'red' | 'indigo';

// API Response Types
export interface PaginatedResponse<T> {
  count: number;
  next?: string;
  previous?: string;
  results: T[];
}

export interface ApiError {
  detail: string;
  status: number;
  code?: string;
  field_errors?: Record<string, string[]>;
}

export interface ApiResponse<T> {
  data?: T;
  error?: ApiError;
  message?: string;
  success: boolean;
}

// Dashboard and Analytics Types
export interface BusinessMetrics {
  total_systems: number;
  active_systems: number;
  total_reports: number;
  critical_alerts: number;
  uptime_percentage: number;
  efficiency_score: number;
  monthly_revenue: number;
  customer_satisfaction: number;

  // Growth indicators
  systems_growth?: number;
  reports_growth?: number;
  revenue_growth?: number;
  satisfaction_growth?: number;
}

export interface ChartData {
  labels: string[];
  datasets: Array<{
    label: string;
    data: number[];
    backgroundColor?: string | string[];
    borderColor?: string | string[];
    borderWidth?: number;
  }>;
}

// Export utility types
export type ID = number | string;
export type Timestamp = string;
export type Currency = number;
export type Percentage = number;
