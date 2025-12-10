/**
 * Application Constants
 * Centralized constants for the application
 */

// =============================================================================
// API Configuration
// =============================================================================

export const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';
export const WS_BASE_URL = process.env.WS_BASE_URL || 'ws://localhost:8000';

export const API_VERSION = 'v1';
export const API_PREFIX = `/api/${API_VERSION}`;

// =============================================================================
// API Endpoints
// =============================================================================

export const API_ENDPOINTS = {
  // Auth
  AUTH: {
    LOGIN: `${API_PREFIX}/auth/login`,
    LOGOUT: `${API_PREFIX}/auth/logout`,
    REFRESH: `${API_PREFIX}/auth/refresh`,
    ME: `${API_PREFIX}/auth/me`,
  },

  // Admin
  ADMIN: {
    METRICS: `${API_PREFIX}/admin/metrics`,
    REVENUE: `${API_PREFIX}/admin/metrics/revenue`,
    TENANTS: `${API_PREFIX}/admin/tenants`,
    USERS: `${API_PREFIX}/admin/users`,
    ALERTS: `${API_PREFIX}/admin/alerts`,
    AUDIT_LOGS: `${API_PREFIX}/admin/audit-logs`,
    PLANS: `${API_PREFIX}/admin/plans/distribution`,
  },

  // Systems
  SYSTEMS: {
    LIST: `${API_PREFIX}/systems`,
    DETAIL: (id: string) => `${API_PREFIX}/systems/${id}`,
    METADATA: (id: string) => `${API_PREFIX}/systems/${id}/metadata`,
    SENSORS: (id: string) => `${API_PREFIX}/systems/${id}/sensors`,
  },

  // Sensors
  SENSORS: {
    LIST: `${API_PREFIX}/sensors`,
    DETAIL: (id: string) => `${API_PREFIX}/sensors/${id}`,
    READINGS: (id: string) => `${API_PREFIX}/sensors/${id}/readings`,
  },

  // Anomalies
  ANOMALIES: {
    LIST: `${API_PREFIX}/anomalies`,
    DETAIL: (id: string) => `${API_PREFIX}/anomalies/${id}`,
    ACKNOWLEDGE: (id: string) => `${API_PREFIX}/anomalies/${id}/acknowledge`,
    RESOLVE: (id: string) => `${API_PREFIX}/anomalies/${id}/resolve`,
  },

  // Reports
  REPORTS: {
    LIST: `${API_PREFIX}/reports`,
    GENERATE: `${API_PREFIX}/reports/generate`,
    DOWNLOAD: (id: string) => `${API_PREFIX}/reports/${id}/download`,
  },
} as const;

// =============================================================================
// WebSocket Endpoints
// =============================================================================

export const WS_ENDPOINTS = {
  ADMIN_METRICS: `${WS_BASE_URL}/ws/admin/metrics`,
  ADMIN_ALERTS: `${WS_BASE_URL}/ws/admin/alerts`,
  SYSTEM_SENSORS: (systemId: string) => `${WS_BASE_URL}/ws/systems/${systemId}/sensors`,
  DIAGNOSIS: (systemId: string) => `${WS_BASE_URL}/ws/diagnosis/${systemId}`,
} as const;

// =============================================================================
// HTTP Status Codes
// =============================================================================

export const HTTP_STATUS = {
  OK: 200,
  CREATED: 201,
  NO_CONTENT: 204,
  BAD_REQUEST: 400,
  UNAUTHORIZED: 401,
  FORBIDDEN: 403,
  NOT_FOUND: 404,
  CONFLICT: 409,
  UNPROCESSABLE_ENTITY: 422,
  INTERNAL_SERVER_ERROR: 500,
  BAD_GATEWAY: 502,
  SERVICE_UNAVAILABLE: 503,
} as const;

// =============================================================================
// Error Codes
// =============================================================================

export const ERROR_CODES = {
  // Auth errors
  AUTH_INVALID_CREDENTIALS: 'AUTH_INVALID_CREDENTIALS',
  AUTH_TOKEN_EXPIRED: 'AUTH_TOKEN_EXPIRED',
  AUTH_TOKEN_INVALID: 'AUTH_TOKEN_INVALID',
  AUTH_UNAUTHORIZED: 'AUTH_UNAUTHORIZED',
  AUTH_FORBIDDEN: 'AUTH_FORBIDDEN',

  // Validation errors
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  VALIDATION_REQUIRED_FIELD: 'VALIDATION_REQUIRED_FIELD',
  VALIDATION_INVALID_FORMAT: 'VALIDATION_INVALID_FORMAT',

  // Resource errors
  RESOURCE_NOT_FOUND: 'RESOURCE_NOT_FOUND',
  RESOURCE_CONFLICT: 'RESOURCE_CONFLICT',
  RESOURCE_DELETED: 'RESOURCE_DELETED',

  // System errors
  SYSTEM_ERROR: 'SYSTEM_ERROR',
  SYSTEM_MAINTENANCE: 'SYSTEM_MAINTENANCE',
  SYSTEM_OVERLOAD: 'SYSTEM_OVERLOAD',
} as const;

// =============================================================================
// UI Constants
// =============================================================================

export const UI = {
  // Breakpoints (matching Tailwind)
  BREAKPOINTS: {
    SM: 640,
    MD: 768,
    LG: 1024,
    XL: 1280,
    '2XL': 1536,
  },

  // Timeouts (ms)
  TIMEOUTS: {
    TOAST: 5000,
    TOAST_ERROR: 0, // Don't auto-dismiss errors
    DEBOUNCE: 300,
    THROTTLE: 100,
    API_REQUEST: 30000,
    WEBSOCKET_RECONNECT: 1000,
  },

  // Animation durations (ms)
  ANIMATIONS: {
    FAST: 150,
    NORMAL: 250,
    SLOW: 500,
  },

  // Z-index layers
  Z_INDEX: {
    BASE: 0,
    DROPDOWN: 1000,
    STICKY: 1020,
    FIXED: 1030,
    MODAL_BACKDROP: 1040,
    MODAL: 1050,
    POPOVER: 1060,
    TOOLTIP: 1070,
    TOAST: 1080,
  },
} as const;

// =============================================================================
// Sensor Configuration
// =============================================================================

export const SENSOR_TYPES = {
  PRESSURE: 'pressure',
  TEMPERATURE: 'temperature',
  FLOW: 'flow',
  VIBRATION: 'vibration',
  POSITION: 'position',
} as const;

export const SENSOR_UNITS = {
  [SENSOR_TYPES.PRESSURE]: ['bar', 'psi', 'Pa', 'MPa'],
  [SENSOR_TYPES.TEMPERATURE]: ['°C', '°F', 'K'],
  [SENSOR_TYPES.FLOW]: ['L/min', 'L/s', 'm³/h', 'gal/min'],
  [SENSOR_TYPES.VIBRATION]: ['mm/s', 'g', 'm/s²'],
  [SENSOR_TYPES.POSITION]: ['mm', 'cm', 'm', 'in'],
} as const;

export const SENSOR_STATUS = {
  ONLINE: 'online',
  OFFLINE: 'offline',
  ERROR: 'error',
  CALIBRATING: 'calibrating',
} as const;

// =============================================================================
// Anomaly Configuration
// =============================================================================

export const ANOMALY_SEVERITY = {
  LOW: 'low',
  MEDIUM: 'medium',
  HIGH: 'high',
  CRITICAL: 'critical',
} as const;

export const ANOMALY_TYPES = {
  PRESSURE_SPIKE: 'pressure_spike',
  TEMPERATURE_HIGH: 'temperature_high',
  FLOW_ANOMALY: 'flow_anomaly',
  VIBRATION_EXCESSIVE: 'vibration_excessive',
  LEAK_DETECTED: 'leak_detected',
  PERFORMANCE_DEGRADATION: 'performance_degradation',
} as const;

// =============================================================================
// User Roles & Permissions
// =============================================================================

export const USER_ROLES = {
  ADMIN: 'admin',
  MANAGER: 'manager',
  OPERATOR: 'operator',
  VIEWER: 'viewer',
} as const;

export const PERMISSIONS = {
  SYSTEMS_READ: 'systems.read',
  SYSTEMS_WRITE: 'systems.write',
  SENSORS_READ: 'sensors.read',
  SENSORS_WRITE: 'sensors.write',
  ANOMALIES_READ: 'anomalies.read',
  ANOMALIES_WRITE: 'anomalies.write',
  ANOMALIES_ACKNOWLEDGE: 'anomalies.acknowledge',
  REPORTS_READ: 'reports.read',
  REPORTS_WRITE: 'reports.write',
  ADMIN_ACCESS: 'admin.access',
} as const;

// =============================================================================
// Plans & Billing
// =============================================================================

export const PLANS = {
  STARTER: 'starter',
  PROFESSIONAL: 'professional',
  ENTERPRISE: 'enterprise',
} as const;

export const PLAN_LIMITS = {
  [PLANS.STARTER]: {
    maxSensors: 10,
    maxUsers: 3,
    features: ['basic_dashboard', 'email_alerts'],
  },
  [PLANS.PROFESSIONAL]: {
    maxSensors: 100,
    maxUsers: 15,
    features: ['advanced_dashboard', 'real_time_monitoring', 'api_access', 'custom_reports'],
  },
  [PLANS.ENTERPRISE]: {
    maxSensors: -1, // Unlimited
    maxUsers: -1, // Unlimited
    features: [
      'all_features',
      'dedicated_support',
      'custom_integration',
      'white_label',
      'sla',
    ],
  },
} as const;

// =============================================================================
// Pagination
// =============================================================================

export const PAGINATION = {
  DEFAULT_PAGE: 1,
  DEFAULT_LIMIT: 20,
  MAX_LIMIT: 100,
} as const;

// =============================================================================
// Cache TTL (ms)
// =============================================================================

export const CACHE_TTL = {
  SHORT: 30000, // 30 seconds
  MEDIUM: 60000, // 1 minute
  LONG: 300000, // 5 minutes
  VERY_LONG: 3600000, // 1 hour
} as const;

// =============================================================================
// Date Formats
// =============================================================================

export const DATE_FORMATS = {
  SHORT: 'short',
  LONG: 'long',
  TIME: 'time',
  DATETIME: 'datetime',
  ISO: 'iso',
} as const;
