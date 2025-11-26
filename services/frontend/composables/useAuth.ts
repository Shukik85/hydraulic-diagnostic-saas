/**
 * Authentication composable
 */

import type { LoginRequest, LoginResponse, UserProfileResponse } from '~/types';

export interface UseAuthReturn {
  accessToken: Ref<string | null>;
  isAuthenticated: ComputedRef<boolean>;
  currentUser: Ref<UserProfileResponse | null>;
  login: (credentials: LoginRequest) => Promise<void>;
  logout: () => Promise<void>;
  refreshToken: () => Promise<void>;
  fetchUserProfile: () => Promise<void>;
}

/**
 * Authentication composable
 */
export const useAuth = (): UseAuthReturn => {
  const authStore = useAuthStore();

  const accessToken = computed(() => authStore.token);
  const isAuthenticated = computed(() => authStore.isAuthenticated);
  const currentUser = computed(() => authStore.user);

  /**
   * Login user
   */
  const login = async (credentials: LoginRequest): Promise<void> => {
    await authStore.login(credentials);
  };

  /**
   * Logout user
   */
  const logout = async (): Promise<void> => {
    await authStore.logout();
  };

  /**
   * Refresh access token
   */
  const refreshToken = async (): Promise<void> => {
    await authStore.refreshToken();
  };

  /**
   * Fetch user profile
   */
  const fetchUserProfile = async (): Promise<void> => {
    await authStore.fetchUserProfile();
  };

  return {
    accessToken,
    isAuthenticated,
    currentUser,
    login,
    logout,
    refreshToken,
    fetchUserProfile,
  };
};
