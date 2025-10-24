// Минимальные типы API
export interface User {
  id: number;
  email: string;
  name: string;
  first_name?: string;
  last_name?: string;
  username?: string;
  role?: string;
  systems_count?: number;
  reports_generated?: number;
  last_activity?: string;
}

export interface HydraulicSystem {
  id: number;
  name: string;
  status: string;
  system_type?: string;
  components_count?: number;
}

export interface DiagnosticReport {
  id: number;
  severity: string;
}

export interface SensorData {
  id: number;
}
export interface RagQueryLog {
  id: number;
}
export interface LoginCredentials {
  email: string;
  password: string;
}
export interface TokenResponse {
  access: string;
  refresh: string;
  user?: User;
}
export interface RegisterData {
  email: string;
  password: string;
}
export interface PaginatedResponse<T> {
  results: T[];
}
export interface ApiError {
  detail: string;
  status: number;
}
