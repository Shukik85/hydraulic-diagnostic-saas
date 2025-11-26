/**
 * Auth Store
 * Authentication state management with Pinia
 */

import { defineStore } from 'pinia';
import type { User, UserRole } from '~/types';

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
  expiresAt: number;
}

interface AuthState {
  user: User | null;
  tokens: AuthTokens | null;
  isLoading: boolean;
  error: string | null;
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    user: null,
    tokens: null,
    isLoading: false,
    error: null,
  }),

  getters: {
    /**
     * Check if user is authenticated
     */
    isAuthenticated: (state): boolean => {
      return !!state.user && !!state.tokens;
    },

    /**
     * Get current user role
     */
    userRole: (state): UserRole | null => {
      return state.user?.role || null;
    },

    /**
     * Check if user is admin
     */
    isAdmin: (state): boolean => {
      return state.user?.role === 'admin';
    },

    /**
     * Check if user is manager or higher
     */
    isManager: (state): boolean => {
      return state.user?.role === 'manager' || state.user?.role === 'admin';
    },

    /**
     * Get user permissions based on role
     */
    permissions: (state): string[] => {
      if (!state.user) return [];

      const rolePermissions: Record<UserRole, string[]> = {
        admin: ['*'], // All permissions
        manager: [
          'systems.read',
          'systems.write',
          'sensors.read',
          'sensors.write',
          'anomalies.read',
          'anomalies.write',
          'reports.read',
          'reports.write',
        ],
        operator: [
          'systems.read',
          'sensors.read',
          'anomalies.read',
          'anomalies.acknowledge',
          'reports.read',
        ],
        viewer: ['systems.read', 'sensors.read', 'anomalies.read', 'reports.read'],
      };

      return rolePermissions[state.user.role] || [];
    },

    /**
     * Check if user has specific permission
     */
    hasPermission:
      (state) =>
      (permission: string): boolean => {
        if (!state.user) return false;
        if (state.user.role === 'admin') return true; // Admin has all permissions

        const userPermissions = useAuthStore().permissions;
        return userPermissions.includes(permission);
      },

    /**
     * Get user display name
     */
    displayName: (state): string => {
      if (!state.user) return '';
      return `${state.user.firstName} ${state.user.lastName}`;
    },
  },

  actions: {
    /**
     * Set user data
     */
    setUser(user: User): void {
      this.user = user;
    },

    /**
     * Set tokens
     */
    setTokens(tokens: AuthTokens): void {
      this.tokens = tokens;
      // Persist to localStorage
      if (typeof window !== 'undefined') {
        localStorage.setItem('auth_tokens', JSON.stringify(tokens));
      }
    },

    /**
     * Clear auth state
     */
    clearAuth(): void {
      this.user = null;
      this.tokens = null;
      this.error = null;
      // Clear from localStorage
      if (typeof window !== 'undefined') {
        localStorage.removeItem('auth_tokens');
      }
    },

    /**
     * Login
     */
    async login(email: string, password: string): Promise<void> {
      this.isLoading = true;
      this.error = null;

      try {
        const api = useApi();
        const response = await api.post<{
          user: User;
          accessToken: string;
          refreshToken: string;
          expiresIn: number;
        }>('/api/v1/auth/login', { email, password });

        const { user, accessToken, refreshToken, expiresIn } = response.data;

        this.setUser(user);
        this.setTokens({
          accessToken,
          refreshToken,
          expiresAt: Date.now() + expiresIn * 1000,
        });
      } catch (error) {
        this.error = error instanceof Error ? error.message : 'Login failed';
        throw error;
      } finally {
        this.isLoading = false;
      }
    },

    /**
     * Logout
     */
    async logout(): Promise<void> {
      try {
        if (this.tokens) {
          const api = useApi();
          await api.post('/api/v1/auth/logout', {
            refreshToken: this.tokens.refreshToken,
          });
        }
      } catch (error) {
        console.error('Logout error:', error);
      } finally {
        this.clearAuth();
      }
    },

    /**
     * Refresh access token
     */
    async refreshToken(): Promise<boolean> {
      if (!this.tokens) return false;

      try {
        const api = useApi();
        const response = await api.post<{
          accessToken: string;
          expiresIn: number;
        }>('/api/v1/auth/refresh', {
          refreshToken: this.tokens.refreshToken,
        });

        const { accessToken, expiresIn } = response.data;

        this.setTokens({
          accessToken,
          refreshToken: this.tokens.refreshToken,
          expiresAt: Date.now() + expiresIn * 1000,
        });

        return true;
      } catch (error) {
        console.error('Token refresh failed:', error);
        this.clearAuth();
        return false;
      }
    },

    /**
     * Load user from token
     */
    async loadUser(): Promise<void> {
      try {
        const api = useApi();
        const response = await api.get<User>('/api/v1/auth/me');
        this.setUser(response.data);
      } catch (error) {
        console.error('Failed to load user:', error);
        this.clearAuth();
      }
    },

    /**
     * Initialize auth from localStorage
     */
    async init(): Promise<void> {
      if (typeof window === 'undefined') return;

      const stored = localStorage.getItem('auth_tokens');
      if (!stored) return;

      try {
        const tokens: AuthTokens = JSON.parse(stored);
        
        // Check if token is expired
        if (Date.now() >= tokens.expiresAt) {
          this.clearAuth();
          return;
        }

        this.tokens = tokens;
        await this.loadUser();
      } catch (error) {
        console.error('Failed to initialize auth:', error);
        this.clearAuth();
      }
    },
  },
});
