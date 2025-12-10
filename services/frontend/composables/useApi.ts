/**
 * API Composable
 * Type-safe API client with retry logic and error handling
 */

import type { ApiResponse, ApiError, RequestConfig } from '~/types';

interface RetryConfig {
  maxRetries: number;
  initialDelay: number;
  maxDelay: number;
  factor: number;
}

const defaultRetryConfig: RetryConfig = {
  maxRetries: 3,
  initialDelay: 1000,
  maxDelay: 10000,
  factor: 2,
};

/**
 * Sleep utility for retry delays
 */
const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Calculate exponential backoff delay
 */
const getRetryDelay = (attempt: number, config: RetryConfig): number => {
  const delay = config.initialDelay * Math.pow(config.factor, attempt);
  return Math.min(delay, config.maxDelay);
};

/**
 * Check if error is retriable
 */
const isRetriableError = (error: unknown): boolean => {
  if (error instanceof Error) {
    // Network errors
    if (error.message.includes('fetch') || error.message.includes('network')) {
      return true;
    }
  }
  
  // HTTP 5xx errors are retriable
  if (typeof error === 'object' && error !== null && 'status' in error) {
    const status = (error as { status: number }).status;
    return status >= 500 && status < 600;
  }
  
  return false;
};

export function useApi() {
  const config = useRuntimeConfig();
  const baseURL = config.public.apiBase || 'http://localhost:8000';

  /**
   * Make API request with retry logic
   */
  async function request<T>(
    endpoint: string,
    options: RequestConfig = {},
    retryConfig: Partial<RetryConfig> = {}
  ): Promise<ApiResponse<T>> {
    const retry = { ...defaultRetryConfig, ...retryConfig };
    let lastError: unknown;

    for (let attempt = 0; attempt <= retry.maxRetries; attempt++) {
      try {
        const response = await $fetch<ApiResponse<T>>(endpoint, {
          baseURL,
          method: options.method || 'GET',
          headers: options.headers,
          params: options.params,
          body: options.body,
          timeout: options.timeout || 30000,
          retry: false, // We handle retry ourselves
        });

        return response;
      } catch (error) {
        lastError = error;

        // Don't retry if not retriable or last attempt
        if (!isRetriableError(error) || attempt === retry.maxRetries) {
          throw error;
        }

        // Wait before retry with exponential backoff
        const delay = getRetryDelay(attempt, retry);
        console.warn(
          `API request failed (attempt ${attempt + 1}/${retry.maxRetries + 1}), retrying in ${delay}ms...`
        );
        await sleep(delay);
      }
    }

    // This should never be reached, but TypeScript needs it
    throw lastError;
  }

  /**
   * GET request
   */
  async function get<T>(
    endpoint: string,
    params?: Record<string, string | number | boolean>
  ): Promise<ApiResponse<T>> {
    return request<T>(endpoint, { method: 'GET', params });
  }

  /**
   * POST request
   */
  async function post<T>(
    endpoint: string,
    body: unknown
  ): Promise<ApiResponse<T>> {
    return request<T>(endpoint, { method: 'POST', body });
  }

  /**
   * PUT request
   */
  async function put<T>(
    endpoint: string,
    body: unknown
  ): Promise<ApiResponse<T>> {
    return request<T>(endpoint, { method: 'PUT', body });
  }

  /**
   * PATCH request
   */
  async function patch<T>(
    endpoint: string,
    body: unknown
  ): Promise<ApiResponse<T>> {
    return request<T>(endpoint, { method: 'PATCH', body });
  }

  /**
   * DELETE request
   */
  async function del<T>(endpoint: string): Promise<ApiResponse<T>> {
    return request<T>(endpoint, { method: 'DELETE' });
  }

  return {
    request,
    get,
    post,
    put,
    patch,
    delete: del,
  };
}
