/**
 * Authentication store
 */

import { defineStore } from 'pinia';
import type { LoginRequest, LoginResponse, UserProfileResponse } from '~/types';

interface AuthState {
  token: string | null;
  refreshToken: string | null;
  user: UserProfileResponse | null;
  isLoading: boolean;
  error: string | null;
}

export const useAuthStore = defineStore('auth', () => {
  // State
  const token = ref<string | null>(null);
  const refreshToken = ref<string | null>(null);
  const user = ref<UserProfileResponse | null>(null);
  const isLoading = ref(false);
  const error = ref<string | null>(null);

  // Getters
  const isAuthenticated = computed(() => !!token.value);
  const userRole = computed(() => user.value?.role || null);
  const isAdmin = computed(() => userRole.value === 'admin');
  const isEngineer = computed(() => userRole.value === 'engineer');

  // Actions
  const login = async (credentials: LoginRequest): Promise<void> => {
    isLoading.value = true;
    error.value = null;

    try {
      const api = useApi();
      const response = await api.post<LoginResponse>('/auth/login', credentials);

      token.value = response.accessToken;
      refreshToken.value = response.refreshToken;

      // Store tokens in localStorage for persistence
      if (process.client) {
        localStorage.setItem('accessToken', response.accessToken);
        localStorage.setItem('refreshToken', response.refreshToken);
      }

      // Fetch user profile
      await fetchUserProfile();
    } catch (err) {
      error.value = err instanceof Error ? err.message : 'Login failed';
      throw err;
    } finally {
      isLoading.value = false;
    }
  };

  const logout = async (): Promise<void> => {
    try {
      const api = useApi();
      await api.post('/auth/logout');
    } catch (err) {
      console.error('Logout error:', err);
    } finally {
      // Clear state
      token.value = null;
      refreshToken.value = null;
      user.value = null;
      error.value = null;

      // Clear localStorage
      if (process.client) {
        localStorage.removeItem('accessToken');
        localStorage.removeItem('refreshToken');
      }

      // Redirect to login
      await navigateTo('/login');
    }
  };

  const refreshAccessToken = async (): Promise<void> => {
    if (!refreshToken.value) {
      throw new Error('No refresh token available');
    }

    try {
      const api = useApi();
      const response = await api.post<LoginResponse>('/auth/refresh', {
        refreshToken: refreshToken.value,
      });

      token.value = response.accessToken;
      refreshToken.value = response.refreshToken;

      // Update localStorage
      if (process.client) {
        localStorage.setItem('accessToken', response.accessToken);
        localStorage.setItem('refreshToken', response.refreshToken);
      }
    } catch (err) {
      // Refresh failed, logout user
      await logout();
      throw err;
    }
  };

  const fetchUserProfile = async (): Promise<void> => {
    try {
      const api = useApi();
      const profile = await api.get<UserProfileResponse>('/auth/profile');
      user.value = profile;
    } catch (err) {
      console.error('Failed to fetch user profile:', err);
      throw err;
    }
  };

  const restoreSession = (): void => {
    if (process.client) {
      const storedToken = localStorage.getItem('accessToken');
      const storedRefreshToken = localStorage.getItem('refreshToken');

      if (storedToken && storedRefreshToken) {
        token.value = storedToken;
        refreshToken.value = storedRefreshToken;

        // Fetch user profile in background
        fetchUserProfile().catch(() => {
          // Session expired, clear tokens
          logout();
        });
      }
    }
  };

  return {
    // State
    token,
    refreshToken,
    user,
    isLoading,
    error,

    // Getters
    isAuthenticated,
    userRole,
    isAdmin,
    isEngineer,

    // Actions
    login,
    logout,
    refreshToken: refreshAccessToken,
    fetchUserProfile,
    restoreSession,
  };
});
