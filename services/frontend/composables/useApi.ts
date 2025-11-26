/**
 * API client composable with retry logic and error handling
 */

import type { ApiError, ApiResponse, PaginatedResponse, RequestConfig } from '~/types';

export interface UseApiReturn {
  get: <T>(url: string, config?: RequestConfig) => Promise<T>;
  post: <T>(url: string, data?: unknown, config?: RequestConfig) => Promise<T>;
  put: <T>(url: string, data?: unknown, config?: RequestConfig) => Promise<T>;
  patch: <T>(url: string, data?: unknown, config?: RequestConfig) => Promise<T>;
  delete: <T>(url: string, config?: RequestConfig) => Promise<T>;
}

/**
 * Custom API error class
 */
export class ApiClientError extends Error {
  constructor(
    public statusCode: number,
    public apiError: ApiError,
    message: string
  ) {
    super(message);
    this.name = 'ApiClientError';
  }
}

/**
 * API client composable
 */
export const useApi = (): UseApiReturn => {
  const config = useRuntimeConfig();
  const { accessToken } = useAuth();

  /**
   * Create fetch options with auth header
   */
  const createOptions = (config?: RequestConfig): RequestInit => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...config?.headers,
    };

    if (accessToken.value) {
      headers.Authorization = `Bearer ${accessToken.value}`;
    }

    return {
      headers,
    };
  };

  /**
   * Handle API errors
   */
  const handleError = async (response: Response): Promise<never> => {
    let errorData: ApiError;
    try {
      errorData = await response.json();
    } catch {
      errorData = {
        error: 'UnknownError',
        message: 'An unknown error occurred',
        statusCode: response.status,
        timestamp: new Date().toISOString(),
      };
    }

    throw new ApiClientError(response.status, errorData, errorData.message);
  };

  /**
   * Make API request with retry logic
   */
  const request = async <T>(
    url: string,
    options: RequestInit,
    config?: RequestConfig
  ): Promise<T> => {
    const fullUrl = `${config.public.apiBase}${url}`;
    const maxRetries = config?.retry ?? 3;
    const retryDelay = config?.retryDelay ?? 1000;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await fetch(fullUrl, options);

        if (!response.ok) {
          // Don't retry on 4xx errors (client errors)
          if (response.status >= 400 && response.status < 500) {
            await handleError(response);
          }

          // Retry on 5xx errors (server errors)
          if (attempt < maxRetries) {
            await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
            continue;
          }

          await handleError(response);
        }

        const data = await response.json();
        return data as T;
      } catch (error) {
        if (error instanceof ApiClientError) {
          throw error;
        }

        if (attempt >= maxRetries) {
          throw error;
        }

        await new Promise((resolve) => setTimeout(resolve, retryDelay * (attempt + 1)));
      }
    }

    throw new Error('Max retries reached');
  };

  return {
    get: async <T>(url: string, config?: RequestConfig): Promise<T> => {
      const options = createOptions(config);
      return request<T>(url, { ...options, method: 'GET' }, config);
    },

    post: async <T>(url: string, data?: unknown, config?: RequestConfig): Promise<T> => {
      const options = createOptions(config);
      return request<T>(
        url,
        {
          ...options,
          method: 'POST',
          body: JSON.stringify(data),
        },
        config
      );
    },

    put: async <T>(url: string, data?: unknown, config?: RequestConfig): Promise<T> => {
      const options = createOptions(config);
      return request<T>(
        url,
        {
          ...options,
          method: 'PUT',
          body: JSON.stringify(data),
        },
        config
      );
    },

    patch: async <T>(url: string, data?: unknown, config?: RequestConfig): Promise<T> => {
      const options = createOptions(config);
      return request<T>(
        url,
        {
          ...options,
          method: 'PATCH',
          body: JSON.stringify(data),
        },
        config
      );
    },

    delete: async <T>(url: string, config?: RequestConfig): Promise<T> => {
      const options = createOptions(config);
      return request<T>(url, { ...options, method: 'DELETE' }, config);
    },
  };
};
