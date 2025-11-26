/**
 * API Client Configuration
 * Setup and configuration for API requests
 */

import type { FetchOptions, FetchResponse } from 'ofetch';
import type { ApiResponse, ApiError } from '~/types';
import { HTTP_STATUS, ERROR_CODES } from './constants';

/**
 * Get auth token from auth store
 */
function getAuthToken(): string | null {
  if (typeof window === 'undefined') return null;
  
  try {
    const stored = localStorage.getItem('auth_tokens');
    if (!stored) return null;
    
    const tokens = JSON.parse(stored);
    return tokens.accessToken || null;
  } catch {
    return null;
  }
}

/**
 * Request interceptor
 * Adds auth headers and common request config
 */
export function onRequest(options: FetchOptions): FetchOptions {
  const token = getAuthToken();

  // Add auth header if token exists
  if (token) {
    options.headers = {
      ...options.headers,
      Authorization: `Bearer ${token}`,
    };
  }

  // Add default headers
  options.headers = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...options.headers,
  };

  // Log request in development
  if (process.env.NODE_ENV === 'development') {
    console.log('[API Request]', options.method || 'GET', options.baseURL, options);
  }

  return options;
}

/**
 * Response interceptor
 * Handles successful responses
 */
export function onResponse<T>(response: FetchResponse<ApiResponse<T>>): ApiResponse<T> {
  // Log response in development
  if (process.env.NODE_ENV === 'development') {
    console.log('[API Response]', response.status, response._data);
  }

  return response._data;
}

/**
 * Error interceptor
 * Maps API errors to application errors
 */
export function onResponseError(error: any): never {
  const { response, request, options } = error;

  // Log error in development
  if (process.env.NODE_ENV === 'development') {
    console.error('[API Error]', {
      status: response?.status,
      url: request?.url || options?.baseURL,
      error: response?._data || error.message,
    });
  }

  // Network error (no response)
  if (!response) {
    throw new Error('Network error: Unable to reach server');
  }

  const status = response.status;
  const data = response._data as ApiError | undefined;

  // Handle specific HTTP status codes
  switch (status) {
    case HTTP_STATUS.UNAUTHORIZED:
      // Token expired or invalid
      if (data?.error?.code === ERROR_CODES.AUTH_TOKEN_EXPIRED) {
        // Trigger token refresh
        // This will be handled by useAuth composable
      }
      throw new Error(data?.error?.message || 'Unauthorized');

    case HTTP_STATUS.FORBIDDEN:
      throw new Error(data?.error?.message || 'Access forbidden');

    case HTTP_STATUS.NOT_FOUND:
      throw new Error(data?.error?.message || 'Resource not found');

    case HTTP_STATUS.UNPROCESSABLE_ENTITY:
      // Validation errors
      throw new Error(data?.error?.message || 'Validation error');

    case HTTP_STATUS.CONFLICT:
      throw new Error(data?.error?.message || 'Resource conflict');

    case HTTP_STATUS.INTERNAL_SERVER_ERROR:
      throw new Error(data?.error?.message || 'Internal server error');

    case HTTP_STATUS.SERVICE_UNAVAILABLE:
      throw new Error('Service temporarily unavailable');

    default:
      throw new Error(data?.error?.message || `Request failed with status ${status}`);
  }
}

/**
 * Retry configuration
 */
export const retryConfig = {
  /**
   * Number of retry attempts
   */
  retries: 3,

  /**
   * Delay between retries (ms)
   */
  retryDelay: 1000,

  /**
   * HTTP status codes to retry
   */
  retryStatusCodes: [
    HTTP_STATUS.INTERNAL_SERVER_ERROR,
    HTTP_STATUS.BAD_GATEWAY,
    HTTP_STATUS.SERVICE_UNAVAILABLE,
  ],

  /**
   * Check if request should be retried
   */
  shouldRetry(response: FetchResponse): boolean {
    return this.retryStatusCodes.includes(response.status);
  },
};

/**
 * Create configured API client
 */
export function createApiClient(baseURL: string) {
  return $fetch.create({
    baseURL,
    retry: retryConfig.retries,
    retryDelay: retryConfig.retryDelay,
    onRequest,
    onResponse,
    onResponseError,
  });
}

/**
 * Default API client instance
 */
export const apiClient = createApiClient(
  process.env.API_BASE_URL || 'http://localhost:8000'
);
