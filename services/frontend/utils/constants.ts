/**
 * Application constants
 */

/**
 * API endpoints
 */
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/auth/login',
    LOGOUT: '/auth/logout',
    REFRESH: '/auth/refresh',
    PROFILE: '/auth/profile',
  },
  SYSTEMS: {
    LIST: '/systems',
    DETAIL: (id: string) => `/systems/${id}`,
    CREATE: '/systems',
    UPDATE: (id: string) => `/systems/${id}`,
    DELETE: (id: string) => `/systems/${id}`,
  },
  SENSORS: {
    LIST: '/sensors',
    DETAIL: (id: string) => `/sensors/${id}`,
    READINGS: (id: string) => `/sensors/${id}/readings`,
  },
  ANOMALIES: {
    LIST: '/anomalies',
    DETAIL: (id: string) => `/anomalies/${id}`,
    ACKNOWLEDGE: (id: string) => `/anomalies/${id}/acknowledge`,
    RESOLVE: (id: string) => `/anomalies/${id}/resolve`,
  },
  ADMIN: {
    METRICS: '/admin/metrics',
    REVENUE: '/admin/metrics/revenue',
    TIERS: '/admin/metrics/tiers',
    TENANTS: '/admin/tenants',
    USERS: '/admin/users',
  },
} as const;

/**
 * WebSocket channels
 */
export const WS_CHANNELS = {
  METRICS: '/admin/metrics',
  SENSORS: (systemId: string) => `/systems/${systemId}/sensors`,
  ANOMALIES: '/anomalies',
} as const;

/**
 * HTTP status codes
 */
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
  SERVICE_UNAVAILABLE: 503,
} as const;

/**
 * Breakpoints for responsive design
 */
export const BREAKPOINTS = {
  SM: 640,
  MD: 768,
  LG: 1024,
  XL: 1280,
  '2XL': 1536,
} as const;

/**
 * Toast durations (milliseconds)
 */
export const TOAST_DURATION = {
  SHORT: 3000,
  MEDIUM: 5000,
  LONG: 7000,
  PERMANENT: 0, // Won't auto-dismiss
} as const;

/**
 * Pagination defaults
 */
export const PAGINATION = {
  DEFAULT_PAGE: 1,
  DEFAULT_LIMIT: 20,
  MAX_LIMIT: 100,
} as const;

/**
 * Date formats
 */
export const DATE_FORMATS = {
  SHORT: 'short',
  LONG: 'long',
  TIME: 'time',
  DATE: 'date',
} as const;

/**
 * Sensor types
 */
export const SENSOR_TYPES = {
  PRESSURE: 'pressure',
  TEMPERATURE: 'temperature',
  FLOW: 'flow',
  VIBRATION: 'vibration',
  POSITION: 'position',
} as const;

/**
 * Anomaly severity levels
 */
export const ANOMALY_SEVERITY = {
  CRITICAL: 'critical',
  HIGH: 'high',
  MEDIUM: 'medium',
  LOW: 'low',
} as const;

/**
 * User roles
 */
export const USER_ROLES = {
  ADMIN: 'admin',
  ENGINEER: 'engineer',
  VIEWER: 'viewer',
} as const;
